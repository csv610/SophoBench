import streamlit as st

from ai2d import AI2DDataset
from blink import BlinkDataset
from cauldron import CauldronDataset
from mathv360k import MathV360KDataset
from mmmu import MMMUDataset
from scienceqa import ScienceQADataset
from worldmedqa import WorldMedQADataset

import base64
from PIL import Image
import io

st.set_page_config(layout="wide")

# Map of dataset names to their handler classes
DATASET_HANDLERS = {
    "AI2 Diagrams": AI2DDataset,
    "BLINK": BlinkDataset,
    "Cauldron": CauldronDataset,
    "MathV360K": MathV360KDataset,
    "MMMU": MMMUDataset,
    "ScienceQA": ScienceQADataset,
    "World MedQA": WorldMedQADataset
}

@st.cache_resource
def get_cached_dataset_handler(dataset_class, model_config):
    return dataset_class(model_config)

@st.cache_data
def get_cached_subjects(dataset_handler):
    return dataset_handler.get_subjects()

@st.cache_data
def get_cached_splits(dataset_handler, subject):
    return dataset_handler.get_splits(subject)

def display_image(image_data):
    """
    Display image from base64 string or file path
    """
    try:
        if isinstance(image_data, str):
            if image_data.startswith('data:image') or image_data.startswith('/9j/'):
                # Handle base64 encoded images
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                st.image(image)
            else:
                # Handle file paths
                st.image(image_data)
        elif isinstance(image_data, bytes):
            # Handle raw bytes
            image = Image.open(io.BytesIO(image_data))
            st.image(image)
        else:
            st.error("Unsupported image format")
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")

def config_panel():
    """
    Configure the sidebar panel for user interaction.
    
    Returns:
        tuple: Contains (dataset_handler, dataset, start_index, end_index) for pagination
    """
    st.sidebar.title("Navigation")

    # Dataset selection
    dataset_name = st.sidebar.selectbox("Select Dataset", list(DATASET_HANDLERS.keys()))
    
    # Model selection
    model_name = st.sidebar.selectbox("Select Model", ["llava1.5", "llava1.6"])
    model_config = {
        'vision': model_name
    }

    # Initialize dataset handler
    dataset_class = DATASET_HANDLERS[dataset_name]
    dataset_handler = get_cached_dataset_handler(dataset_class, model_config)
        
    subjects = get_cached_subjects(dataset_handler)
    if not subjects:
        st.sidebar.error("No subjects available")
        return None, None, 0, 0

    # Subset selection using dataset's partitions
    subject = st.sidebar.selectbox("Select Subset", subjects)
        
    splits = get_cached_splits(dataset_handler, subject)
    if not splits:
        st.sidebar.error("No splits available for selected subject")
        return None, None, 0, 0

    # Split selection
    split = st.sidebar.selectbox("Select Split", splits)

    # Load dataset
    dataset = dataset_handler.get_dataset(subject, split)
    if dataset and len(dataset) > 0:
        # Select number of items per page
        num_items_per_page = st.sidebar.slider("Items per Page", min_value=1, max_value=5, value=3)
            
        # Calculate total pages
        total_items = len(dataset)
        total_pages = (total_items + num_items_per_page - 1) // num_items_per_page

        # Page selection box
        page_options = list(range(1, total_pages + 1))
        current_page = st.sidebar.selectbox("Select Page", page_options)

        # Calculate start and end indices for the current page
        start_index = (current_page - 1) * num_items_per_page
        end_index = min(start_index + num_items_per_page, total_items)

        return dataset_handler, dataset, start_index, end_index
    else:
        st.sidebar.error("No data available for selected split")
        return None, None, 0, 0

def view_dataset():
    """
    Display the dataset in a paginated format with images, questions and choices.
    Handles the main UI layout and interaction logic.
    """
    st.title("Visual Dataset Viewer")
    
    # Get configuration and dataset
    dataset_handler, dataset, start_index, end_index = config_panel()
    
    if dataset is None:
        return

    # Display items for the current page
    for idx in range(start_index, end_index):
        with st.expander(f"Question {idx + 1}", expanded=True):
            item = dataset[idx]
            
            # Display image if available
            if 'image' in item:
                st.markdown("**Image:**")
                display_image(item['image'])
            
            # Display question
            st.markdown("**Question:**")
            st.write(item.get('question', 'No question available'))
            
            # Display choices
            st.markdown("**Choices:**")
            choices = item.get('options', [])
            if choices:
                for i, choice in enumerate(choices):
                    st.write(f"{chr(65 + i)}. {choice}")
            
            # Display answer
            if st.button(f"Show Answer {idx + 1}"):
                st.markdown("**Answer:**")
                answer = item.get('answer')
                if isinstance(answer, int):
                    st.write(f"Option {chr(65 + answer)}")
                else:
                    st.write(answer)

if __name__ == "__main__":
    view_dataset()
