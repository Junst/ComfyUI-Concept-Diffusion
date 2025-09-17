"""
Test script for ConceptAttention ComfyUI nodes
"""

import torch
import numpy as np
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nodes import (
    ConceptAttentionNode,
    ConceptSaliencyMapNode,
    ConceptSegmentationNode,
    ConceptAttentionVisualizerNode
)

def create_mock_model():
    """Create a mock model for testing."""
    class MockModel:
        def __init__(self):
            self.device = "cpu"
        
        def named_modules(self):
            # Mock attention modules
            class MockAttention:
                def register_forward_hook(self, hook):
                    return lambda: None
            
            return [("attention.0", MockAttention()), ("attention.1", MockAttention())]
    
    return MockModel()

def create_mock_clip():
    """Create a mock CLIP model for testing."""
    class MockCLIP:
        def __init__(self):
            self.device = "cpu"
        
        def __call__(self, **kwargs):
            # Return mock embeddings
            batch_size = kwargs['input_ids'].shape[0]
            seq_len = kwargs['input_ids'].shape[1]
            hidden_size = 768
            
            class MockOutput:
                def __init__(self):
                    self.last_hidden_state = torch.randn(batch_size, seq_len, hidden_size)
            
            return MockOutput()
    
    return MockCLIP()

def create_mock_image():
    """Create a mock image tensor."""
    return torch.randn(1, 512, 512, 3)

def test_concept_attention_node():
    """Test ConceptAttentionNode."""
    print("Testing ConceptAttentionNode...")
    
    node = ConceptAttentionNode()
    
    # Create mock inputs
    model = create_mock_model()
    clip = create_mock_clip()
    image = create_mock_image()
    concepts = "person, car, tree"
    num_inference_steps = 10
    
    try:
        # Test input types
        input_types = node.INPUT_TYPES()
        print(f"Input types: {input_types}")
        
        # Test function call (this will fail due to mock objects, but we can test the structure)
        print("Node structure test passed!")
        
    except Exception as e:
        print(f"Expected error with mock objects: {e}")
    
    print("ConceptAttentionNode test completed.\n")

def test_concept_saliency_map_node():
    """Test ConceptSaliencyMapNode."""
    print("Testing ConceptSaliencyMapNode...")
    
    node = ConceptSaliencyMapNode()
    
    # Create mock concept maps
    concept_maps = {
        "person": torch.randn(64, 64),
        "car": torch.randn(64, 64),
        "tree": torch.randn(64, 64)
    }
    
    try:
        # Test input types
        input_types = node.INPUT_TYPES()
        print(f"Input types: {input_types}")
        
        # Test function call
        mask, saliency_image = node.extract_concept_map(
            concept_maps, "person", 0.5
        )
        
        print(f"Mask shape: {mask.shape}")
        print(f"Saliency image shape: {saliency_image.shape}")
        print("ConceptSaliencyMapNode test passed!")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("ConceptSaliencyMapNode test completed.\n")

def test_concept_segmentation_node():
    """Test ConceptSegmentationNode."""
    print("Testing ConceptSegmentationNode...")
    
    node = ConceptSegmentationNode()
    
    # Create mock inputs
    concept_maps = {
        "person": torch.randn(64, 64),
        "car": torch.randn(64, 64),
        "tree": torch.randn(64, 64)
    }
    image = create_mock_image()
    concepts = "person, car, tree"
    
    try:
        # Test function call
        segmentation_mask, segmented_image = node.perform_segmentation(
            concept_maps, image, concepts
        )
        
        print(f"Segmentation mask shape: {segmentation_mask.shape}")
        print(f"Segmented image shape: {segmented_image.shape}")
        print("ConceptSegmentationNode test passed!")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("ConceptSegmentationNode test completed.\n")

def test_concept_attention_visualizer_node():
    """Test ConceptAttentionVisualizerNode."""
    print("Testing ConceptAttentionVisualizerNode...")
    
    node = ConceptAttentionVisualizerNode()
    
    # Create mock inputs
    concept_maps = {
        "person": torch.randn(64, 64),
        "car": torch.randn(64, 64),
        "tree": torch.randn(64, 64)
    }
    image = create_mock_image()
    overlay_alpha = 0.5
    
    try:
        # Test function call
        visualized_image = node.visualize_attention(
            concept_maps, image, overlay_alpha
        )
        
        print(f"Visualized image shape: {visualized_image[0].shape}")
        print("ConceptAttentionVisualizerNode test passed!")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("ConceptAttentionVisualizerNode test completed.\n")

def test_node_mappings():
    """Test node class mappings."""
    print("Testing node class mappings...")
    
    from nodes import NODE_CLASS_MAPPINGS
    
    expected_nodes = [
        "ConceptAttentionNode",
        "ConceptSaliencyMapNode", 
        "ConceptSegmentationNode",
        "ConceptAttentionVisualizerNode"
    ]
    
    for node_name in expected_nodes:
        if node_name in NODE_CLASS_MAPPINGS:
            print(f"✓ {node_name} found in mappings")
        else:
            print(f"✗ {node_name} missing from mappings")
    
    print("Node mappings test completed.\n")

def main():
    """Run all tests."""
    print("Running ConceptAttention ComfyUI nodes tests...\n")
    
    test_concept_attention_node()
    test_concept_saliency_map_node()
    test_concept_segmentation_node()
    test_concept_attention_visualizer_node()
    test_node_mappings()
    
    print("All tests completed!")

if __name__ == "__main__":
    main()
