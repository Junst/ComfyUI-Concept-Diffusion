"""
Simplified ComfyUI nodes for Concept Attention based on original structure.
"""

import torch
import numpy as np
from PIL import Image
import logging
from typing import Dict, List, Tuple, Any
from .concept_attention import ConceptAttentionProcessor

def _ensure_image_hw3(arr, target_hw=None):
    """
    arr: (H,W,3) 또는 (H,W) 또는 (N,3) 또는 (N,) 등 각종 입력을 받아
         최종 (H,W,3) float32 [0..1] 로 변환.
    """
    a = np.asarray(arr)

    # 1) torch.Tensor -> numpy
    try:
        import torch
        if isinstance(arr, torch.Tensor):
            a = arr.detach().float().cpu().numpy()
    except Exception:
        pass

    # 2) (N,3) 또는 (N,)이면 정사각형으로 복원
    if a.ndim == 2 and a.shape[1] in (1, 3):  # (N,1) or (N,3)
        N = a.shape[0]
        side = int(np.sqrt(N))
        if side * side != N:
            # 제곱수로 패딩
            next_side = int(np.ceil(np.sqrt(N)))
            pad = next_side * next_side - N
            pad_axis = ((0, pad), (0, 0))
            a = np.pad(a, pad_axis, mode="constant")
            side = next_side
        a = a.reshape(side, side, a.shape[1])

    elif a.ndim == 1:  # (N,)
        N = a.shape[0]
        side = int(np.sqrt(N))
        if side * side != N:
            next_side = int(np.ceil(np.sqrt(N)))
            pad = next_side * next_side - N
            a = np.pad(a, (0, pad), mode="constant")
            side = next_side
        a = a.reshape(side, side)

    # 3) (H,W) → (H,W,3) 그레이스케일 확장
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)

    # 4) (H,W,1) → (H,W,3)
    if a.ndim == 3 and a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)

    # 5) 타입/스케일 정규화
    a = a.astype(np.float32)
    mn, mx = float(a.min()), float(a.max())
    if mx > mn:
        a = (a - mn) / (mx - mn)
    else:
        a[:] = 0.0

    # 6) 필요 시 리사이즈
    if target_hw is not None:
        H, W = target_hw
        a_u8 = (a * 255.0).astype(np.uint8)
        a_u8 = np.array(Image.fromarray(a_u8).resize((W, H), Image.BILINEAR))
        if a_u8.ndim == 2:
            a_u8 = np.stack([a_u8]*3, axis=-1)
        a = a_u8.astype(np.float32) / 255.0

    # 최종 보장: (H,W,3) float32 0..1
    return a

def to_preview_image(arr_1d_or_2d, target_hw=None):
    """Legacy function - use _ensure_image_hw3 instead"""
    return _ensure_image_hw3(arr_1d_or_2d, target_hw)

logger = logging.getLogger(__name__)

class ConceptAttentionNode:
    """
    Simplified ComfyUI node for concept attention.
    Based on original ConceptAttention structure.
    """
    
    RETURN_TYPES = ("CONCEPT_MAPS", "IMAGE")
    RETURN_NAMES = ("concept_maps", "visualized_image")
    FUNCTION = "generate_concept_attention"
    CATEGORY = "Concept Attention"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful landscape"}),
                "concept_list": ("STRING", {"multiline": True, "default": "woman, cat, white, lines, cane"}),
            }
        }
    
    def __init__(self):
        self.processor = None
        
    def generate_concept_attention(self, model, clip, image, prompt, concept_list):
        """
        Generate concept attention maps using simplified approach.
        """
        try:
            # Convert concept_list string to list
            if isinstance(concept_list, str):
                concept_list = [concept.strip() for concept in concept_list.split(',')]
            elif isinstance(concept_list, list):
                concept_list = [str(concept).strip() for concept in concept_list]
            else:
                concept_list = []
            
            logger.info(f"DEBUG: Extracted concepts from prompt: {concept_list}")
            
            # Initialize processor if not exists
            if self.processor is None:
                self.processor = ConceptAttentionProcessor(model, device="cuda")
            
            # Process image to get concept maps
            concept_maps = self.processor.process_image(image, concept_list, clip)
            
            # Convert to ComfyUI format with guaranteed (H,W,3) output
            image_shape = image.shape if hasattr(image, 'shape') else None
            saliency_maps, visualized_image = self._convert_to_comfyui_format(concept_maps, image_shape)
            
            logger.info(f"DEBUG: processor.process_image returned: {type(concept_maps)}")
            logger.info(f"DEBUG: saliency_maps keys: {list(saliency_maps.keys()) if saliency_maps else None}")
            logger.info(f"DEBUG: visualized_image shape: {visualized_image.shape}")
            
            return saliency_maps, visualized_image
            
        except Exception as e:
            logger.error(f"Error in SimpleConceptAttentionNode: {e}")
            # Return empty results on error
            return {}, image
    
    def _convert_to_comfyui_format(self, concept_maps: Dict[str, torch.Tensor], image_shape=None) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Convert concept maps to ComfyUI format with guaranteed (H,W,3) output.
        Returns: (fixed_maps, visualized_image)
        """
        try:
            if not concept_maps:
                logger.warning("WARNING: saliency_maps is empty, returning empty ConceptMaps")
                # Create fallback image
                fallback_img = _ensure_image_hw3(np.zeros((1024, 1024)), target_hw=(1024, 1024))
                return {}, fallback_img
            
            # Get target dimensions from image shape
            if image_shape is not None and len(image_shape) >= 2:
                target_h, target_w = image_shape[:2]
            else:
                target_h, target_w = 1024, 1024
            
            # Convert all concept maps to guaranteed (H,W,3) format
            fixed_maps = {}
            for concept, attention_map in concept_maps.items():
                try:
                    fixed_maps[concept] = _ensure_image_hw3(attention_map, target_hw=(target_h, target_w))
                except Exception as e:
                    logger.error(f"Error processing concept '{concept}': {e}")
                    # Create fallback
                    fixed_maps[concept] = _ensure_image_hw3(np.zeros((32, 32)), target_hw=(target_h, target_w))
            
            # Create visualized image from first concept map
            if fixed_maps:
                visualized_image = next(iter(fixed_maps.values()))  # (H,W,3) float32
            else:
                visualized_image = _ensure_image_hw3(np.zeros((target_h, target_w)), target_hw=(target_h, target_w))
            
            logger.info(f"DEBUG: Converted concept_maps with keys: {list(fixed_maps.keys())}")
            logger.info(f"DEBUG: visualized_image shape: {visualized_image.shape}")
            
            return fixed_maps, visualized_image
            
        except Exception as e:
            logger.error(f"Error converting concept maps: {e}")
            fallback_img = _ensure_image_hw3(np.zeros((1024, 1024)), target_hw=(1024, 1024))
            return {}, fallback_img
    
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
                if len(img_np.shape) == 3 and img_np.shape[0] == 3:  # CHW format
                    img_np = np.transpose(img_np, (1, 2, 0))
            else:
                img_np = np.array(image)
            
            # Ensure image is in correct format
            if len(img_np.shape) == 2:  # Grayscale
                img_np = np.stack([img_np] * 3, axis=-1)
            elif len(img_np.shape) == 3 and img_np.shape[-1] == 1:  # Single channel
                img_np = np.repeat(img_np, 3, axis=-1)
            
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
                    
                    # Ensure attention_map is numpy array
                    if isinstance(attention_map, torch.Tensor):
                        attention_map = attention_map.cpu().numpy()
                    
                    # Use _ensure_image_hw3 for robust conversion
                    preview_img = _ensure_image_hw3(attention_map, target_hw=(img_np.shape[0], img_np.shape[1]))
                    
                    # Use the first channel as attention map
                    attention_resized = preview_img[:, :, 0]
                    
                    # Normalize back to 0-1 range
                    attention_resized = attention_resized.astype(np.float32) / 255.0
                    
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


class ConceptSaliencyMapNode:
    """
    Node for extracting saliency maps from concept attention.
    """
    
    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "saliency_image")
    FUNCTION = "extract_saliency_map"
    CATEGORY = "Concept Attention"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "concept_maps": ("CONCEPT_MAPS",),
                "concept_name": ("STRING", {"default": "woman"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    def __init__(self):
        pass
    
    def extract_saliency_map(self, concept_maps, concept_name, threshold=0.5):
        """
        Extract saliency map for a specific concept.
        """
        try:
            if not concept_maps or concept_name not in concept_maps:
                logger.warning(f"Concept '{concept_name}' not found in concept_maps")
                return None, None
            
            concept_map = concept_maps[concept_name]
            
            # Apply threshold
            saliency_map = (concept_map > threshold).float()
            
            # Create visualization
            saliency_image = self._create_saliency_visualization(concept_map, threshold)
            
            return saliency_map, saliency_image
            
        except Exception as e:
            logger.error(f"Error extracting saliency map: {e}")
            return None, None
    
    def _create_saliency_visualization(self, concept_map, threshold):
        """
        Create visualization of saliency map.
        """
        try:
            # Normalize concept map
            normalized_map = (concept_map - concept_map.min()) / (concept_map.max() - concept_map.min() + 1e-8)
            
            # Apply threshold
            thresholded_map = (normalized_map > threshold).float()
            
            # Create colored visualization
            visualization = torch.stack([
                thresholded_map,  # Red channel
                torch.zeros_like(thresholded_map),  # Green channel
                torch.zeros_like(thresholded_map)   # Blue channel
            ], dim=0)
            
            return visualization
            
        except Exception as e:
            logger.error(f"Error creating saliency visualization: {e}")
            return None


class ConceptSegmentationNode:
    """
    Node for performing segmentation based on concept attention.
    """
    
    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("segmentation_mask", "segmented_image")
    FUNCTION = "perform_segmentation"
    CATEGORY = "Concept Attention"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "concept_maps": ("CONCEPT_MAPS",),
                "image": ("IMAGE",),
                "concepts": ("STRING", {"multiline": True, "default": "woman, cat, white, lines, cane"}),
            }
        }
    
    def __init__(self):
        pass
    
    def perform_segmentation(self, concept_maps, image, concepts):
        """
        Perform segmentation based on concept maps.
        """
        try:
            if not concept_maps:
                logger.warning("No concept maps available for segmentation")
                return None, image
            
            # Create segmentation mask
            segmentation_mask = torch.zeros_like(image[0, 0])  # Single channel mask
            
            for i, concept in enumerate(concepts):
                if concept in concept_maps:
                    concept_map = concept_maps[concept]
                    # Assign different values for different concepts
                    segmentation_mask += concept_map * (i + 1)
            
            # Normalize segmentation mask
            if segmentation_mask.max() > 0:
                segmentation_mask = segmentation_mask / segmentation_mask.max()
            
            # Create segmented image
            segmented_image = self._create_segmented_image(image, segmentation_mask)
            
            return segmentation_mask.unsqueeze(0).unsqueeze(0), segmented_image
            
        except Exception as e:
            logger.error(f"Error in segmentation: {e}")
            return None, image
    
    def _create_segmented_image(self, image, segmentation_mask):
        """
        Create segmented image visualization.
        """
        try:
            # Convert to numpy if needed
            if isinstance(image, torch.Tensor):
                img_np = image.squeeze().cpu().numpy()
                if img_np.shape[0] == 3:  # CHW format
                    img_np = np.transpose(img_np, (1, 2, 0))
            else:
                img_np = np.array(image)
            
            # Normalize image
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            
            # Apply segmentation mask
            segmented = img_np * segmentation_mask.cpu().numpy()
            
            # Convert back to tensor
            if len(segmented.shape) == 3:
                segmented = np.transpose(segmented, (2, 0, 1))  # HWC to CHW
            
            return torch.from_numpy(segmented).unsqueeze(0)
            
        except Exception as e:
            logger.error(f"Error creating segmented image: {e}")
            return image


class ConceptAttentionVisualizerNode:
    """
    Node for visualizing concept attention maps.
    """
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("visualized_image",)
    FUNCTION = "visualize_attention"
    CATEGORY = "Concept Attention"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "concept_maps": ("CONCEPT_MAPS",),
                "image": ("IMAGE",),
            }
        }
    
    def __init__(self):
        pass
    
    def visualize_attention(self, concept_maps, image):
        """
        Create visualization of concept attention maps.
        """
        try:
            if not concept_maps:
                logger.warning("No concept maps available for visualization")
                return image
            
            # Convert image to numpy if needed
            if isinstance(image, torch.Tensor):
                img_np = image.squeeze().cpu().numpy()
                if img_np.shape[0] == 3:  # CHW format
                    img_np = np.transpose(img_np, (1, 2, 0))
            else:
                img_np = np.array(image)
            
            # Normalize image
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
            
            for i, (concept, attention_map) in enumerate(concept_maps.items()):
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
            logger.error(f"Error creating attention visualization: {e}")
            return image


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "ConceptAttentionNode": ConceptAttentionNode,
    "ConceptSaliencyMapNode": ConceptSaliencyMapNode,
    "ConceptSegmentationNode": ConceptSegmentationNode,
    "ConceptAttentionVisualizerNode": ConceptAttentionVisualizerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConceptAttentionNode": "Concept Attention",
    "ConceptSaliencyMapNode": "Concept Saliency Map",
    "ConceptSegmentationNode": "Concept Segmentation",
    "ConceptAttentionVisualizerNode": "Concept Attention Visualizer",
}
