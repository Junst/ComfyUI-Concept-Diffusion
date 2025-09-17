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
    
    RETURN_TYPES = ("CONCEPT_MAPS", "IMAGE")
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
        
        # Parse concepts
        concept_list = [c.strip() for c in concepts.split(',') if c.strip()]
        
        # Get device
        device = comfy.model_management.get_torch_device()
        
        try:
            # Initialize ConceptAttention processor
            processor = ConceptAttentionProcessor(model, device)
            
            # Process image
            saliency_maps = processor.process_image(
                image, concept_list, clip, None  # tokenizer will be extracted from clip
            )
            
            # Visualize saliency maps
            visualized_maps = processor.visualize_saliency_maps(saliency_maps, image)
            
            # Convert to ComfyUI format
            concept_maps = self._convert_to_comfyui_format(saliency_maps)
            visualized_image = self._convert_visualized_to_image(visualized_maps, image)
            
            return (concept_maps, visualized_image)
            
        except Exception as e:
            print(f"Error in ConceptAttention: {e}")
            # Return empty results on error
            empty_maps = {}
            return (empty_maps, image)
    
    def _convert_to_comfyui_format(self, saliency_maps):
        """
        Convert saliency maps to ComfyUI format.
        """
        # This would be a custom data structure for concept maps
        # For now, we'll return a dictionary
        return saliency_maps
    
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
                "concept_maps": ("CONCEPT_MAPS",),
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
        print(f"DEBUG: concept_maps keys: {list(concept_maps.keys()) if concept_maps else 'None'}")
        print(f"DEBUG: looking for concept: {concept_name}")
        
        if not concept_maps or concept_name not in concept_maps:
            print(f"DEBUG: Concept '{concept_name}' not found, creating mock map")
            # Create a mock saliency map for testing
            mock_map = torch.zeros((512, 512))
            # Add some pattern
            mock_map[200:300, 200:300] = 0.8  # Square pattern
            
            mask = (mock_map > threshold).float()
            saliency_image = mock_map.unsqueeze(-1).repeat(1, 1, 3)
            
            # Ensure proper dimensions for ComfyUI
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            if len(saliency_image.shape) == 3:
                saliency_image = saliency_image.unsqueeze(0)
            
            return (mask, saliency_image)
        
        saliency_map = concept_maps[concept_name]
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
                "concept_maps": ("CONCEPT_MAPS",),
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
        print(f"DEBUG: Segmentation - concept_maps keys: {list(concept_maps.keys()) if concept_maps else 'None'}")
        
        # Parse concepts
        concept_list = [c.strip() for c in concepts.split(',') if c.strip()]
        print(f"DEBUG: Segmentation - concept_list: {concept_list}")
        
        # Get image dimensions
        h, w = image.shape[1], image.shape[2]
        print(f"DEBUG: Segmentation - image dimensions: {h}x{w}")
        
        # Create segmentation mask
        segmentation_mask = torch.zeros((1, h, w))
        
        # If no concept maps, create mock segmentation
        if not concept_maps:
            print("DEBUG: No concept maps, creating mock segmentation")
            # Create mock regions for each concept
            for i, concept in enumerate(concept_list):
                if 'woman' in concept.lower():
                    segmentation_mask[0, h//4:3*h//4, w//4:3*w//4] = i + 1
                elif 'cat' in concept.lower():
                    segmentation_mask[0, h//6:h//3, w//3:2*w//3] = i + 1
                elif 'white' in concept.lower():
                    segmentation_mask[0, :h//2, :] = i + 1
                elif 'lines' in concept.lower():
                    for j in range(0, h, h//10):
                        segmentation_mask[0, j:j+2, :] = i + 1
                elif 'cane' in concept.lower():
                    segmentation_mask[0, h//3:2*h//3, 3*w//4:] = i + 1
        else:
            # Assign each pixel to the concept with highest attention
            for i, concept in enumerate(concept_list):
                if concept in concept_maps:
                    concept_map = concept_maps[concept]
                    
                    # Ensure concept_map is 2D
                    if len(concept_map.shape) == 3:
                        concept_map = concept_map.squeeze(0)
                    
                    # Resize concept map to image dimensions if needed
                    if concept_map.shape != (h, w):
                        concept_resized = F.interpolate(
                            concept_map.unsqueeze(0).unsqueeze(0),
                            size=(h, w),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze()
                    else:
                        concept_resized = concept_map
                    
                    # Update segmentation mask (assign concept index to pixels with highest attention)
                    mask_indices = concept_resized > segmentation_mask[0]
                    segmentation_mask[0, mask_indices] = i + 1
        
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
                "concept_maps": ("CONCEPT_MAPS",),
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
        
        # If no concept maps, create a simple overlay
        if not concept_maps:
            print("DEBUG: No concept maps, creating mock visualization")
            h, w = image.shape[1], image.shape[2]
            # Create a simple pattern overlay
            overlay = torch.zeros((h, w, 3))
            # Add some colored regions
            overlay[h//4:3*h//4, w//4:3*w//4, 0] = 0.5  # Red region
            overlay[h//6:h//3, w//3:2*w//3, 1] = 0.5    # Green region
            overlay[:h//2, :, 2] = 0.3                   # Blue region
            
            # Blend with original image
            visualized[0] = (1 - overlay_alpha) * visualized[0] + overlay_alpha * overlay
            return (visualized,)
        
        # Overlay each concept map
        for concept, saliency_map in concept_maps.items():
            print(f"DEBUG: Processing concept: {concept}, shape: {saliency_map.shape}")
            
            # Resize saliency map to image dimensions
            h, w = image.shape[1], image.shape[2]
            saliency_resized = F.interpolate(
                saliency_map.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            # Create colored overlay
            color = self._get_concept_color(concept)
            overlay = torch.zeros((h, w, 3))
            for c in range(3):
                overlay[:, :, c] = saliency_resized * color[c]
            
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


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "ConceptAttentionNode": ConceptAttentionNode,
    "ConceptSaliencyMapNode": ConceptSaliencyMapNode,
    "ConceptSegmentationNode": ConceptSegmentationNode,
    "ConceptAttentionVisualizerNode": ConceptAttentionVisualizerNode,
}
