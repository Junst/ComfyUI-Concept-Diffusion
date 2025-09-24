"""
Simplified ComfyUI nodes for Concept Attention based on original structure.
"""

import torch
import numpy as np
from PIL import Image
import logging
from typing import Dict, List, Tuple, Any
from .concept_attention_simple import SimpleConceptAttentionProcessor

logger = logging.getLogger(__name__)

class SimpleConceptAttentionNode:
    """
    Simplified ComfyUI node for concept attention.
    Based on original ConceptAttention structure.
    """
    
    def __init__(self):
        self.processor = None
        
    def generate_concept_attention(self, model, clip, image, prompt, concept_list):
        """
        Generate concept attention maps using simplified approach.
        """
        try:
            logger.info(f"DEBUG: Extracted concepts from prompt: {concept_list}")
            
            # Initialize processor if not exists
            if self.processor is None:
                self.processor = SimpleConceptAttentionProcessor(model, device="cuda")
            
            # Process image to get concept maps
            concept_maps = self.processor.process_image(image, concept_list, clip)
            
            # Convert to ComfyUI format
            saliency_maps = self._convert_to_comfyui_format(concept_maps)
            
            # Create visualization
            visualized_image = self._create_simple_visualization(saliency_maps, image)
            
            logger.info(f"DEBUG: processor.process_image returned: {type(concept_maps)}")
            logger.info(f"DEBUG: saliency_maps keys: {list(saliency_maps.keys()) if saliency_maps else None}")
            
            return saliency_maps, visualized_image
            
        except Exception as e:
            logger.error(f"Error in SimpleConceptAttentionNode: {e}")
            # Return empty results on error
            return {}, image
    
    def _convert_to_comfyui_format(self, concept_maps: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Convert concept maps to ComfyUI format.
        """
        try:
            if not concept_maps:
                logger.warning("WARNING: saliency_maps is empty, returning empty ConceptMaps")
                return {}
            
            # Simple conversion - just return the concept maps
            saliency_maps = {}
            for concept, attention_map in concept_maps.items():
                # Ensure tensor is on CPU and convert to numpy
                if isinstance(attention_map, torch.Tensor):
                    attention_map = attention_map.cpu().numpy()
                
                # Normalize to 0-1 range
                attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
                
                saliency_maps[concept] = attention_map
            
            logger.info(f"DEBUG: Converted concept_maps with keys: {list(saliency_maps.keys())}")
            return saliency_maps
            
        except Exception as e:
            logger.error(f"Error converting to ComfyUI format: {e}")
            return {}
    
    def _create_simple_visualization(self, saliency_maps: Dict[str, torch.Tensor], 
                                   image: torch.Tensor) -> torch.Tensor:
        """
        Create simple visualization of concept attention maps.
        """
        try:
            if not saliency_maps:
                logger.warning("WARNING: saliency_maps is empty, returning empty ConceptMaps")
                return image
            
            # Convert image to numpy if needed
            if isinstance(image, torch.Tensor):
                img_np = image.squeeze().cpu().numpy()
                if img_np.shape[0] == 3:  # CHW format
                    img_np = np.transpose(img_np, (1, 2, 0))
            else:
                img_np = np.array(image)
            
            # Normalize image to 0-1 range
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            
            # Create overlay for each concept
            overlay = img_np.copy()
            colors = [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green
                [0.0, 0.0, 1.0],  # Blue
                [1.0, 1.0, 0.0],  # Yellow
                [1.0, 0.0, 1.0],  # Magenta
            ]
            
            for i, (concept, attention_map) in enumerate(saliency_maps.items()):
                if i < len(colors):
                    color = colors[i]
                    # Resize attention map to image size
                    attention_resized = np.array(Image.fromarray(attention_map).resize(
                        (img_np.shape[1], img_np.shape[0]), Image.BILINEAR
                    ))
                    
                    # Create colored overlay
                    for c in range(3):
                        overlay[:, :, c] += attention_resized * color[c] * 0.3
            
            # Clip to valid range
            overlay = np.clip(overlay, 0, 1)
            
            # Convert back to tensor
            if len(overlay.shape) == 3:
                overlay = np.transpose(overlay, (2, 0, 1))  # HWC to CHW
            
            return torch.from_numpy(overlay).unsqueeze(0)
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return image
    
    def extract_concept_map(self, concept_maps: Dict[str, torch.Tensor], 
                          concept_name: str) -> torch.Tensor:
        """
        Extract specific concept map.
        """
        try:
            if not concept_maps:
                raise ValueError(f"Concept '{concept_name}' not found in concept_maps. Available: None")
            
            if concept_name not in concept_maps:
                available = list(concept_maps.keys()) if concept_maps else "None"
                raise ValueError(f"Concept '{concept_name}' not found in concept_maps. Available: {available}")
            
            return torch.from_numpy(concept_maps[concept_name])
            
        except Exception as e:
            logger.error(f"Error extracting concept map: {e}")
            raise ValueError(f"Failed to extract concept map for '{concept_name}': {e}")
    
    def perform_segmentation(self, concept_maps: Dict[str, torch.Tensor], 
                           concept_list: List[str], image: torch.Tensor) -> torch.Tensor:
        """
        Perform segmentation based on concept maps.
        """
        try:
            if not concept_maps:
                logger.warning("WARNING: saliency_maps is empty, returning empty ConceptMaps")
                return image
            
            # Create segmentation mask
            segmentation_mask = torch.zeros_like(image[0, 0])  # Single channel mask
            
            for i, concept in enumerate(concept_list):
                if concept in concept_maps:
                    concept_map = torch.from_numpy(concept_maps[concept])
                    # Assign different values for different concepts
                    segmentation_mask += concept_map * (i + 1)
            
            # Normalize segmentation mask
            if segmentation_mask.max() > 0:
                segmentation_mask = segmentation_mask / segmentation_mask.max()
            
            return segmentation_mask.unsqueeze(0).unsqueeze(0)
            
        except Exception as e:
            logger.error(f"Error in segmentation: {e}")
            return image


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "SimpleConceptAttentionNode": SimpleConceptAttentionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleConceptAttentionNode": "Simple Concept Attention",
}
