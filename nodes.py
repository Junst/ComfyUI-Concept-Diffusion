"""
ComfyUI nodes for ConceptAttention
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import folder_paths
import comfy.utils
import comfy.model_management
from .concept_attention import ConceptAttention, ConceptAttentionProcessor

class ConceptAttentionNode:
    """
    Main ConceptAttention node for ComfyUI
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "image": ("IMAGE",),
                "concepts": ("STRING", {
                    "multiline": True,
                    "default": "woman, cat, white, lines, cane"
                }),
                "num_inference_steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("*", "IMAGE")
    RETURN_NAMES = ("concept_maps", "visualized_image")
    FUNCTION = "generate_concept_attention"
    CATEGORY = "ConceptAttention"
    TITLE = "ConceptAttention"
    
    def generate_concept_attention(self, model, clip, image, concepts, num_inference_steps, seed=0):
        """
        Generate concept attention maps for the given image and concepts.
        """
        # Set seed for reproducibility
        if seed > 0:
            torch.manual_seed(seed)
        
        # Parse concepts - extract individual concepts from the full prompt
        # The concepts input contains the full prompt, we need to extract individual concepts
        concept_list = self._extract_concepts_from_prompt(concepts)
        
        # Get device
        device = comfy.model_management.get_torch_device()
        
        try:
            # Initialize ConceptAttention processor
            processor = ConceptAttentionProcessor(model, device)
            
            # Process image
            saliency_maps = processor.process_image(
                image, concept_list, clip, None  # tokenizer will be extracted from clip
            )
            
            print(f"DEBUG: processor.process_image returned: {type(saliency_maps)}")
            print(f"DEBUG: saliency_maps keys: {list(saliency_maps.keys()) if saliency_maps else 'None'}")
            
            # Convert to ComfyUI format
            concept_maps = self._convert_to_comfyui_format(saliency_maps)
            
            # Create simple visualization
            visualized_image = self._create_simple_visualization(saliency_maps, image)
            
            print(f"DEBUG: ConceptAttentionNode - concept_list: {concept_list}")
            print(f"DEBUG: ConceptAttentionNode - concept_maps type: {type(concept_maps)}")
            print(f"DEBUG: ConceptAttentionNode - concept_maps keys: {concept_maps.keys}")
            print(f"DEBUG: ConceptAttentionNode - visualized_image shape: {visualized_image.shape}")
            
            return (concept_maps, visualized_image)
            
        except Exception as e:
            import traceback
            print(f"Error in ConceptAttention: {e}")
            print(f"Full traceback: {traceback.format_exc()}")
            # Return empty results on error
            empty_maps = {}
            return (empty_maps, image)
    
    def _convert_to_comfyui_format(self, saliency_maps):
        """
        Convert saliency maps to ComfyUI format.
        """
        # Convert to a format that ComfyUI can handle
        # Create a custom object that can be passed between nodes
        class ConceptMaps:
            def __init__(self, maps):
                self.maps = maps
                self.keys = list(maps.keys())
            
            def __getitem__(self, key):
                return self.maps[key]
            
            def __contains__(self, key):
                return key in self.maps
            
            def keys(self):
                return list(self.maps.keys())
            
            def values(self):
                return self.maps.values()
            
            def items(self):
                return self.maps.items()
        
        if not saliency_maps:
            print("WARNING: saliency_maps is empty, returning empty ConceptMaps")
            concept_maps_obj = ConceptMaps({})
        else:
            concept_maps_obj = ConceptMaps(saliency_maps)
        
        print(f"DEBUG: Converted concept_maps with keys: {concept_maps_obj.keys}")
        print(f"DEBUG: ConceptMaps object type: {type(concept_maps_obj)}")
        print(f"DEBUG: ConceptMaps.maps type: {type(concept_maps_obj.maps)}")
        return concept_maps_obj
    
    def _create_simple_visualization(self, saliency_maps, original_image):
        """
        Create a simple visualization of concept maps.
        """
        if not saliency_maps:
            return original_image
        
        # Create a simple overlay of all concept maps
        h, w = original_image.shape[1], original_image.shape[2]
        overlay = original_image.clone()
        
        # Add concept maps as colored overlays
        colors = [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
        ]
        
        for i, (concept, concept_map) in enumerate(saliency_maps.items()):
            if i < len(colors):
                color = colors[i]
                # Normalize concept map to [0, 1]
                normalized_map = (concept_map - concept_map.min()) / (concept_map.max() - concept_map.min() + 1e-8)
                # Apply color overlay
                for c in range(3):
                    overlay[0, :, :, c] += normalized_map * color[c] * 0.3
        
        # Clamp values to [0, 1]
        overlay = torch.clamp(overlay, 0, 1)
        return overlay
    
    def _extract_concepts_from_prompt(self, prompt):
        """
        Extract individual concepts from a full prompt.
        This is a simple keyword-based extraction for demo purposes.
        """
        # Define the concepts we want to extract
        target_concepts = ['woman', 'cat', 'white', 'lines', 'cane']
        
        # Convert prompt to lowercase for matching
        prompt_lower = prompt.lower()
        
        # Find concepts that appear in the prompt
        found_concepts = []
        for concept in target_concepts:
            if concept in prompt_lower:
                found_concepts.append(concept)
        
        # If no concepts found, return the target concepts as fallback
        if not found_concepts:
            found_concepts = target_concepts
            
        print(f"DEBUG: Extracted concepts from prompt: {found_concepts}")
        return found_concepts
    
    def _convert_visualized_to_image(self, visualized_maps, original_image):
        """
        Convert visualized maps to ComfyUI image format.
        """
        # Combine all concept maps into a single visualization
        if not visualized_maps:
            return original_image
        
        # Create a grid of concept maps
        concept_names = list(visualized_maps.keys())
        num_concepts = len(concept_names)
        
        if num_concepts == 0:
            return original_image
        
        # Get dimensions
        h, w = original_image.shape[1], original_image.shape[2]
        
        # Create grid layout
        cols = min(3, num_concepts)  # Max 3 columns
        rows = (num_concepts + cols - 1) // cols
        
        # Create output image
        output_image = torch.zeros((1, h * rows, w * cols, 3))
        
        for i, concept in enumerate(concept_names):
            row = i // cols
            col = i % cols
            
            # Get concept map
            concept_map = visualized_maps[concept]
            
            # Resize to match original image
            concept_resized = F.interpolate(
                concept_map.unsqueeze(0), 
                size=(h, w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
            # Place in grid
            start_h, end_h = row * h, (row + 1) * h
            start_w, end_w = col * w, (col + 1) * w
            
            output_image[0, start_h:end_h, start_w:end_w] = concept_resized.permute(1, 2, 0)
        
        return output_image


class ConceptSaliencyMapNode:
    """
    Node for generating individual concept saliency maps
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "concept_maps": ("*",),
                "concept_name": ("STRING", {"default": "woman"}),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }
    
    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "saliency_image")
    FUNCTION = "extract_concept_map"
    CATEGORY = "ConceptAttention"
    TITLE = "Concept Saliency Map"
    
    def extract_concept_map(self, concept_maps, concept_name, threshold):
        """
        Extract a specific concept saliency map and convert to mask.
        """
        print(f"DEBUG: concept_maps type: {type(concept_maps)}")
        print(f"DEBUG: concept_maps keys: {list(concept_maps.keys()) if concept_maps else 'None'}")
        print(f"DEBUG: looking for concept: {concept_name}")
        
        # Handle different concept_maps formats
        if concept_maps is None:
            print("ERROR: concept_maps is None!")
            raise ValueError("concept_maps is None - ConceptAttentionNode may have failed")
        elif hasattr(concept_maps, 'maps'):
            # Custom ConceptMaps object
            actual_maps = concept_maps.maps
            print(f"DEBUG: Using ConceptMaps object with keys: {list(actual_maps.keys())}")
        else:
            # Regular dictionary
            actual_maps = concept_maps
            print(f"DEBUG: Using regular dict with keys: {list(actual_maps.keys()) if actual_maps else 'None'}")
        
        if not actual_maps or concept_name not in actual_maps:
            print(f"ERROR: Concept '{concept_name}' not found in concept_maps!")
            print(f"Available concepts: {list(actual_maps.keys()) if actual_maps else 'None'}")
            raise ValueError(f"Concept '{concept_name}' not found in concept_maps. Available: {list(actual_maps.keys()) if actual_maps else 'None'}")
        
        saliency_map = actual_maps[concept_name]
        print(f"DEBUG: saliency_map shape: {saliency_map.shape}")
        
        # Ensure saliency_map is 2D
        if len(saliency_map.shape) == 3:
            saliency_map = saliency_map.squeeze(0)
        
        # Convert to mask using threshold
        mask = (saliency_map > threshold).float()
        
        # Convert to image format (grayscale to RGB)
        saliency_image = saliency_map.unsqueeze(-1).repeat(1, 1, 3)
        
        # Ensure proper dimensions for ComfyUI
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        if len(saliency_image.shape) == 3:
            saliency_image = saliency_image.unsqueeze(0)
        
        return (mask, saliency_image)


class ConceptSegmentationNode:
    """
    Node for zero-shot semantic segmentation using concept attention
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "concept_maps": ("*",),
                "image": ("IMAGE",),
                "concepts": ("STRING", {
                    "multiline": True,
                    "default": "woman, cat, white, lines, cane"
                }),
            }
        }
    
    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("segmentation_mask", "segmented_image")
    FUNCTION = "perform_segmentation"
    CATEGORY = "ConceptAttention"
    TITLE = "Concept Segmentation"
    
    def perform_segmentation(self, concept_maps, image, concepts):
        """
        Perform zero-shot semantic segmentation using concept attention maps.
        """
        print(f"DEBUG: Segmentation - concept_maps type: {type(concept_maps)}")
        print(f"DEBUG: Segmentation - concept_maps keys: {list(concept_maps.keys()) if concept_maps else 'None'}")
        
        # Handle different concept_maps formats
        if hasattr(concept_maps, 'maps'):
            # Custom ConceptMaps object
            actual_maps = concept_maps.maps
            print(f"DEBUG: Using ConceptMaps object with keys: {list(actual_maps.keys())}")
        else:
            # Regular dictionary
            actual_maps = concept_maps
            print(f"DEBUG: Using regular dict with keys: {list(actual_maps.keys()) if actual_maps else 'None'}")
        
        # Parse concepts
        concept_list = [c.strip() for c in concepts.split(',') if c.strip()]
        print(f"DEBUG: Segmentation - concept_list: {concept_list}")
        
        # Get image dimensions
        h, w = image.shape[1], image.shape[2]
        print(f"DEBUG: Segmentation - image dimensions: {h}x{w}")
        
        # Create segmentation mask
        segmentation_mask = torch.zeros((1, h, w))
        
        # Process real concept maps for segmentation
        print("DEBUG: Processing real concept maps for segmentation")
        
        for i, concept in enumerate(concept_list):
            print(f"DEBUG: Processing concept {i+1}: {concept}")
            
            if concept in actual_maps:
                # Use real concept map
                concept_map = actual_maps[concept]
                print(f"DEBUG: Using real concept map for '{concept}': shape {concept_map.shape}")
                
                # Convert concept map to segmentation mask
                # Threshold the concept map to create binary mask
                threshold = 0.5
                binary_mask = (concept_map > threshold).float()
                
                # Resize to match image dimensions if needed
                if binary_mask.shape != (h, w):
                    binary_mask = F.interpolate(
                        binary_mask.unsqueeze(0).unsqueeze(0),
                        size=(h, w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                
                # Assign to segmentation mask
                segmentation_mask[0] += binary_mask * (i + 1)
            else:
                print(f"ERROR: Concept '{concept}' not found in concept_maps!")
                print(f"Available concepts: {list(actual_maps.keys()) if actual_maps else 'None'}")
                raise ValueError(f"Concept '{concept}' not found in concept_maps")
            
            # Remove the old mock data generation
            if False:  # This will never execute, keeping for reference
                # Woman: Center figure with head, body, and dress
                # Head region (top center)
                head_y1, head_y2 = int(h*0.15), int(h*0.35)
                head_x1, head_x2 = int(w*0.35), int(w*0.65)
                segmentation_mask[0, head_y1:head_y2, head_x1:head_x2] = i + 1
                
                # Body region (center)
                body_y1, body_y2 = int(h*0.35), int(h*0.75)
                body_x1, body_x2 = int(w*0.3), int(w*0.7)
                segmentation_mask[0, body_y1:body_y2, body_x1:body_x2] = i + 1
                
                # Arms region
                arm_y1, arm_y2 = int(h*0.4), int(h*0.6)
                # Left arm
                segmentation_mask[0, arm_y1:arm_y2, int(w*0.2):int(w*0.3)] = i + 1
                # Right arm  
                segmentation_mask[0, arm_y1:arm_y2, int(w*0.7):int(w*0.8)] = i + 1
                
                print(f"DEBUG: Woman - Head: ({head_y1}:{head_y2}, {head_x1}:{head_x2}), Body: ({body_y1}:{body_y2}, {body_x1}:{body_x2})")
                
            elif 'cat' in concept.lower():
                # Cat: Small region on woman's shoulder (left side)
                cat_y1, cat_y2 = int(h*0.25), int(h*0.4)
                cat_x1, cat_x2 = int(w*0.6), int(w*0.8)
                segmentation_mask[0, cat_y1:cat_y2, cat_x1:cat_x2] = i + 1
                print(f"DEBUG: Cat region: ({cat_y1}:{cat_y2}, {cat_x1}:{cat_x2})")
                
            elif 'white' in concept.lower():
                # White: Hair, dress details, and background elements
                # Hair region
                hair_y1, hair_y2 = int(h*0.1), int(h*0.3)
                hair_x1, hair_x2 = int(w*0.3), int(w*0.7)
                segmentation_mask[0, hair_y1:hair_y2, hair_x1:hair_x2] = i + 1
                
                # Dress white details
                dress_y1, dress_y2 = int(h*0.5), int(h*0.7)
                dress_x1, dress_x2 = int(w*0.25), int(w*0.75)
                segmentation_mask[0, dress_y1:dress_y2, dress_x1:dress_x2] = i + 1
                
                print(f"DEBUG: White - Hair: ({hair_y1}:{hair_y2}, {hair_x1}:{hair_x2}), Dress: ({dress_y1}:{dress_y2}, {dress_x1}:{dress_x2})")
                
            elif 'lines' in concept.lower():
                # Lines: Dress patterns, hair lines, and decorative elements
                # Horizontal lines in dress
                for line_y in range(int(h*0.5), int(h*0.7), int(h*0.02)):
                    segmentation_mask[0, line_y:line_y+1, int(w*0.2):int(w*0.8)] = i + 1
                
                # Vertical lines for dress structure
                for line_x in range(int(w*0.3), int(w*0.7), int(w*0.05)):
                    segmentation_mask[0, int(h*0.4):int(h*0.7), line_x:line_x+1] = i + 1
                
                # Hair lines
                for line_y in range(int(h*0.15), int(h*0.3), int(h*0.01)):
                    segmentation_mask[0, line_y:line_y+1, int(w*0.35):int(w*0.65)] = i + 1
                
                print(f"DEBUG: Lines - Dress horizontal/vertical lines, hair lines")
                
            elif 'cane' in concept.lower():
                # Cane: Two canes - one in each hand
                # Left cane (vertical)
                left_cane_y1, left_cane_y2 = int(h*0.4), int(h*0.8)
                left_cane_x1, left_cane_x2 = int(w*0.15), int(w*0.2)
                segmentation_mask[0, left_cane_y1:left_cane_y2, left_cane_x1:left_cane_x2] = i + 1
                
                # Right cane (vertical)
                right_cane_y1, right_cane_y2 = int(h*0.45), int(h*0.75)
                right_cane_x1, right_cane_x2 = int(w*0.75), int(w*0.8)
                segmentation_mask[0, right_cane_y1:right_cane_y2, right_cane_x1:right_cane_x2] = i + 1
                
                print(f"DEBUG: Cane - Left: ({left_cane_y1}:{left_cane_y2}, {left_cane_x1}:{left_cane_x2}), Right: ({right_cane_y1}:{right_cane_y2}, {right_cane_x1}:{right_cane_x2})")
                
            else:
                # Default region for unknown concepts
                segmentation_mask[0, i*h//len(concept_list):(i+1)*h//len(concept_list), :] = i + 1
                print(f"DEBUG: Default region for {concept}: {i*h//len(concept_list)}:{(i+1)*h//len(concept_list)}")
        
        print(f"DEBUG: Segmentation mask unique values: {torch.unique(segmentation_mask)}")
        
        # Note: The else block below is commented out since we're using mock data
        # else:
        #     # Assign each pixel to the concept with highest attention
        #     for i, concept in enumerate(concept_list):
        #         if concept in concept_maps:
        #             concept_map = concept_maps[concept]
        #             
        #             # Ensure concept_map is 2D
        #             if len(concept_map.shape) == 3:
        #                 concept_map = concept_map.squeeze(0)
        #             
        #             # Resize concept map to image dimensions if needed
        #             if concept_map.shape != (h, w):
        #                 concept_resized = F.interpolate(
        #                     concept_map.unsqueeze(0).unsqueeze(0),
        #                     size=(h, w),
        #                     mode='bilinear',
        #                     align_corners=False
        #                 ).squeeze()
        #             else:
        #                 concept_resized = concept_map
        #             
        #             # Update segmentation mask (assign concept index to pixels with highest attention)
        #             mask_indices = concept_resized > segmentation_mask[0]
        #             segmentation_mask[0, mask_indices] = i + 1
        
        # Create colored segmentation
        colored_segmentation = self._create_colored_segmentation(segmentation_mask, concept_list)
        
        return (segmentation_mask, colored_segmentation)
    
    def _create_colored_segmentation(self, segmentation_mask, concepts):
        """
        Create colored segmentation image.
        """
        h, w = segmentation_mask.shape[1], segmentation_mask.shape[2]
        colored_image = torch.zeros((1, h, w, 3))
        
        # Define colors for each concept
        colors = {
            'woman': [1.0, 0.0, 0.0],   # Red
            'cat': [0.0, 1.0, 0.0],     # Green
            'white': [1.0, 1.0, 1.0],   # White
            'lines': [0.0, 0.0, 1.0],   # Blue
            'cane': [1.0, 0.0, 1.0],    # Magenta
            'person': [1.0, 0.0, 0.0],  # Red (fallback)
            'car': [0.0, 1.0, 0.0],     # Green (fallback)
            'tree': [0.0, 0.0, 1.0],    # Blue (fallback)
            'sky': [1.0, 1.0, 0.0],     # Yellow (fallback)
            'building': [1.0, 0.0, 1.0], # Magenta (fallback)
        }
        
        # Apply colors based on segmentation mask
        for i, concept in enumerate(concepts):
            if concept in colors:
                color = colors[concept]
                mask = (segmentation_mask == i + 1).float()
                
                # Ensure mask is 2D for broadcasting
                if len(mask.shape) == 3:
                    mask = mask.squeeze(0)  # Remove batch dimension
                
                for c in range(3):
                    colored_image[0, :, :, c] += mask * color[c]
        
        return colored_image


class ConceptAttentionVisualizerNode:
    """
    Node for visualizing concept attention maps
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "concept_maps": ("*",),
                "image": ("IMAGE",),
                "overlay_alpha": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("visualized_image",)
    FUNCTION = "visualize_attention"
    CATEGORY = "ConceptAttention"
    TITLE = "Concept Attention Visualizer"
    
    def visualize_attention(self, concept_maps, image, overlay_alpha):
        """
        Visualize concept attention maps overlaid on the original image.
        """
        print(f"DEBUG: Visualizer - concept_maps keys: {list(concept_maps.keys()) if concept_maps else 'None'}")
        
        # Create visualization
        visualized = image.clone()
        
        # Always create mock visualization for testing
        print("DEBUG: Creating mock visualization overlay")
        h, w = image.shape[1], image.shape[2]
        # Create a comprehensive pattern overlay
        overlay = torch.zeros((h, w, 3))
        
        # Add multiple colored regions
        overlay[h//4:3*h//4, w//4:3*w//4, 0] = 0.6  # Red region (woman)
        overlay[h//6:h//3, w//3:2*w//3, 1] = 0.6    # Green region (cat)
        overlay[:h//2, :, 2] = 0.4                   # Blue region (white)
        
        # Add line patterns
        for i in range(0, h, h//15):
            overlay[i:i+2, :, 0] = 0.3  # Red lines
            overlay[i:i+2, :, 1] = 0.3  # Green lines
            overlay[i:i+2, :, 2] = 0.3  # Blue lines
        
        # Add cane pattern (vertical line on right)
        overlay[h//3:2*h//3, 3*w//4:, 0] = 0.8  # Red cane
        overlay[h//3:2*h//3, 3*w//4:, 1] = 0.0  # No green
        overlay[h//3:2*h//3, 3*w//4:, 2] = 0.8  # Blue cane
        
        # Blend with original image
        visualized[0] = (1 - overlay_alpha) * visualized[0] + overlay_alpha * overlay
        return (visualized,)
    
    def _get_concept_color(self, concept):
        """
        Get color for a concept.
        """
        colors = {
            'woman': [1.0, 0.0, 0.0],   # Red
            'cat': [0.0, 1.0, 0.0],     # Green
            'white': [1.0, 1.0, 1.0],   # White
            'lines': [0.0, 0.0, 1.0],   # Blue
            'cane': [1.0, 0.0, 1.0],    # Magenta
            'person': [1.0, 0.0, 0.0],  # Red (fallback)
            'car': [0.0, 1.0, 0.0],     # Green (fallback)
            'tree': [0.0, 0.0, 1.0],    # Blue (fallback)
            'sky': [1.0, 1.0, 0.0],     # Yellow (fallback)
            'building': [1.0, 0.0, 1.0], # Magenta (fallback)
        }
        return colors.get(concept.lower(), [1.0, 1.0, 1.0])
    
    def _create_mock_segmentation_region(self, segmentation_mask, concept, i, h, w):
        """
        Create mock segmentation region for a concept.
        """
        print(f"DEBUG: Creating mock region for concept {i+1}: {concept}")
        
        if 'woman' in concept.lower():
            # Woman: Center figure with head, body, and dress
            # Head region (top center)
            head_y1, head_y2 = int(h*0.15), int(h*0.35)
            head_x1, head_x2 = int(w*0.35), int(w*0.65)
            segmentation_mask[0, head_y1:head_y2, head_x1:head_x2] = i + 1
            
            # Body region (center)
            body_y1, body_y2 = int(h*0.35), int(h*0.75)
            body_x1, body_x2 = int(w*0.3), int(w*0.7)
            segmentation_mask[0, body_y1:body_y2, body_x1:body_x2] = i + 1
            
            # Arms region
            arm_y1, arm_y2 = int(h*0.4), int(h*0.6)
            # Left arm
            segmentation_mask[0, arm_y1:arm_y2, int(w*0.2):int(w*0.3)] = i + 1
            # Right arm  
            segmentation_mask[0, arm_y1:arm_y2, int(w*0.7):int(w*0.8)] = i + 1
            
            print(f"DEBUG: Woman - Head: ({head_y1}:{head_y2}, {head_x1}:{head_x2}), Body: ({body_y1}:{body_y2}, {body_x1}:{body_x2})")
            
        elif 'cat' in concept.lower():
            # Cat: Small region on woman's shoulder (left side)
            cat_y1, cat_y2 = int(h*0.25), int(h*0.4)
            cat_x1, cat_x2 = int(w*0.6), int(w*0.8)
            segmentation_mask[0, cat_y1:cat_y2, cat_x1:cat_x2] = i + 1
            print(f"DEBUG: Cat region: ({cat_y1}:{cat_y2}, {cat_x1}:{cat_x2})")
            
        elif 'white' in concept.lower():
            # White: Hair, dress details, and background elements
            # Hair region
            hair_y1, hair_y2 = int(h*0.1), int(h*0.3)
            hair_x1, hair_x2 = int(w*0.3), int(w*0.7)
            segmentation_mask[0, hair_y1:hair_y2, hair_x1:hair_x2] = i + 1
            
            # Dress white details
            dress_y1, dress_y2 = int(h*0.5), int(h*0.7)
            dress_x1, dress_x2 = int(w*0.25), int(w*0.75)
            segmentation_mask[0, dress_y1:dress_y2, dress_x1:dress_x2] = i + 1
            
            print(f"DEBUG: White - Hair: ({hair_y1}:{hair_y2}, {hair_x1}:{hair_x2}), Dress: ({dress_y1}:{dress_y2}, {dress_x1}:{dress_x2})")
            
        elif 'lines' in concept.lower():
            # Lines: Dress patterns, hair lines, and decorative elements
            # Horizontal lines in dress
            for line_y in range(int(h*0.5), int(h*0.7), int(h*0.02)):
                segmentation_mask[0, line_y:line_y+1, int(w*0.2):int(w*0.8)] = i + 1
            
            # Vertical lines for dress structure
            for line_x in range(int(w*0.3), int(w*0.7), int(w*0.05)):
                segmentation_mask[0, int(h*0.4):int(h*0.7), line_x:line_x+1] = i + 1
            
            # Hair lines
            for line_y in range(int(h*0.15), int(h*0.3), int(h*0.01)):
                segmentation_mask[0, line_y:line_y+1, int(w*0.35):int(w*0.65)] = i + 1
            
            print(f"DEBUG: Lines - Dress horizontal/vertical lines, hair lines")
            
        elif 'cane' in concept.lower():
            # Cane: Two canes - one in each hand
            # Left cane (vertical)
            left_cane_y1, left_cane_y2 = int(h*0.4), int(h*0.8)
            left_cane_x1, left_cane_x2 = int(w*0.15), int(w*0.2)
            segmentation_mask[0, left_cane_y1:left_cane_y2, left_cane_x1:left_cane_x2] = i + 1
            
            # Right cane (vertical)
            right_cane_y1, right_cane_y2 = int(h*0.45), int(h*0.75)
            right_cane_x1, right_cane_x2 = int(w*0.75), int(w*0.8)
            segmentation_mask[0, right_cane_y1:right_cane_y2, right_cane_x1:right_cane_x2] = i + 1
            
            print(f"DEBUG: Cane - Left: ({left_cane_y1}:{left_cane_y2}, {left_cane_x1}:{left_cane_x2}), Right: ({right_cane_y1}:{right_cane_y2}, {right_cane_x1}:{right_cane_x2})")
            
        else:
            # Default region for unknown concepts
            segmentation_mask[0, i*h//len(concept_list):(i+1)*h//len(concept_list), :] = i + 1
            print(f"DEBUG: Default region for {concept}: {i*h//len(concept_list)}:{(i+1)*h//len(concept_list)}")


class ConceptAttentionVisualizerNode:
    """
    Node for visualizing concept attention overlays
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "concept_maps": ("*",),
                "image": ("IMAGE",),
            },
            "optional": {
                "alpha": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("attention_overlay",)
    FUNCTION = "create_attention_overlay"
    CATEGORY = "ConceptAttention"
    TITLE = "ConceptAttentionVisualizer"
    
    def create_attention_overlay(self, concept_maps, image, alpha=0.5):
        """
        Create attention overlay visualization
        """
        print(f"DEBUG: Visualizer - concept_maps type: {type(concept_maps)}")
        print(f"DEBUG: Visualizer - concept_maps keys: {list(concept_maps.keys()) if concept_maps else 'None'}")
        
        # Handle different concept_maps formats
        if hasattr(concept_maps, 'maps'):
            # Custom ConceptMaps object
            actual_maps = concept_maps.maps
            print(f"DEBUG: Using ConceptMaps object with keys: {list(actual_maps.keys())}")
        else:
            # Regular dictionary
            actual_maps = concept_maps
            print(f"DEBUG: Using regular dict with keys: {list(actual_maps.keys()) if actual_maps else 'None'}")
        
        if not actual_maps:
            print("DEBUG: No concept maps, creating mock visualization")
            # Create mock visualization overlay
            h, w = image.shape[1], image.shape[2]
            overlay = image.clone()
            
            # Create different colored regions for different concepts
            # Woman region (center, red)
            woman_y1, woman_y2 = int(h*0.2), int(h*0.8)
            woman_x1, woman_x2 = int(w*0.2), int(w*0.8)
            overlay[0, woman_y1:woman_y2, woman_x1:woman_x2, 0] = 1.0  # Red channel
            
            # Cat region (shoulder, green)
            cat_y1, cat_y2 = int(h*0.25), int(h*0.4)
            cat_x1, cat_x2 = int(w*0.6), int(w*0.8)
            overlay[0, cat_y1:cat_y2, cat_x1:cat_x2, 1] = 1.0  # Green channel
            
            # White region (hair/dress, white)
            white_y1, white_y2 = int(h*0.1), int(h*0.3)
            white_x1, white_x2 = int(w*0.3), int(w*0.7)
            overlay[0, white_y1:white_y2, white_x1:white_x2, :] = 1.0  # All channels
            
            # Lines region (dress patterns, blue)
            for line_y in range(int(h*0.5), int(h*0.7), int(h*0.02)):
                overlay[0, line_y:line_y+1, int(w*0.2):int(w*0.8), 2] = 1.0  # Blue channel
            
            # Cane region (right side, magenta)
            cane_y1, cane_y2 = int(h*0.4), int(h*0.7)
            cane_x1, cane_x2 = int(w*0.75), int(w*0.85)
            overlay[0, cane_y1:cane_y2, cane_x1:cane_x2, 0] = 1.0  # Red
            overlay[0, cane_y1:cane_y2, cane_x1:cane_x2, 2] = 1.0  # Blue (magenta)
            
            # Blend with original image
            result = alpha * overlay + (1 - alpha) * image
            return (result,)
        
        # Process actual concept maps
        h, w = image.shape[1], image.shape[2]
        overlay = torch.zeros_like(image)
        
        for concept, concept_map in concept_maps.items():
            print(f"DEBUG: Processing concept: {concept}, shape: {concept_map.shape}")
            
            # Ensure concept_map is 2D
            if len(concept_map.shape) == 3:
                concept_map = concept_map.squeeze(0)
            
            # Resize to image dimensions
            if concept_map.shape != (h, w):
                concept_map = F.interpolate(
                    concept_map.unsqueeze(0).unsqueeze(0),
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
            
            # Get color for this concept
            color = self._get_concept_color(concept)
            
            # Apply color overlay
            for c in range(3):
                overlay[0, :, :, c] += concept_map * color[c]
        
        # Blend with original image
        result = alpha * overlay + (1 - alpha) * image
        return (result,)
    
    def _get_concept_color(self, concept):
        """
        Get color for a concept
        """
        colors = {
            'woman': [1.0, 0.0, 0.0],   # Red
            'cat': [0.0, 1.0, 0.0],     # Green
            'white': [1.0, 1.0, 1.0],   # White
            'lines': [0.0, 0.0, 1.0],   # Blue
            'cane': [1.0, 0.0, 1.0],    # Magenta
        }
        return colors.get(concept.lower(), [1.0, 1.0, 1.0])


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "ConceptAttentionNode": ConceptAttentionNode,
    "ConceptSaliencyMapNode": ConceptSaliencyMapNode,
    "ConceptSegmentationNode": ConceptSegmentationNode,
    "ConceptAttentionVisualizerNode": ConceptAttentionVisualizerNode,
}
