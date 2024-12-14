import streamlit as st
from gpqa import GPQADataset
from gsm8k import GSM8KDataset
from gsmplus import GSMPlusDataset
from imo_geometry import IMOGeometryDataset
from math import MathDataset
from medicalquestions import MedicalQuestionsDataset
from medicationqa import MedicationQADataset
from medical_meadow_flashcards import MedicalMeadowFlashcardsDataset
from medical_meadow_wikidoc_patient import MedicalMeadowWikidocPatientDataset
from metamathqa import MetaMathQADataset
from metamathqa40k import MetaMathQA40KDataset
from truthfulqa import TruthfulQADataset
from scibench import SciBenchDataset

st.set_page_config(layout="wide")

# Map of dataset names to their handler classes
DATASET_HANDLERS = {
    "GSM8K": GSM8KDataset,
    "GSM Plus": GSMPlusDataset,
    "Math": MathDataset,
    "MetaMathQA": MetaMathQADataset,
    "MetaMathQA 40K": MetaMathQA40KDataset,
    "IMO Geometry": IMOGeometryDataset,
    "GPQA": GPQADataset,
    "TruthfulQA": TruthfulQADataset,
    "SciBench": SciBenchDataset,
    "SimpleQA": SimpleQADataset,
    "Medical Questions": MedicalQuestionsDataset,
    "Medication QA": MedicationQADataset,
    "Medical Meadow Flashcards": MedicalMeadowFlashcardsDataset,
    "Medical Meadow Wikidoc Patient": MedicalMeadowWikidocPatientDataset,
    "NEJM Llama": NEJMLlamaDataset
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
    model_name = st.sidebar.selectbox("Select Model", ["llama3.2", "llama3.1"])
    model_config = {
        'text': model_name
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
        num_items_per_page = st.sidebar.slider("Items per Page", min_value=1, max_value=10, value=5)
            
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
    Display the dataset in a paginated format with questions and answers.
    Handles the main UI layout and interaction logic.
    """
    st.title("Dataset Viewer")
    
    # Get configuration and dataset
    dataset_handler, dataset, start_index, end_index = config_panel()
    
    if dataset is None:
        return

    # Display items for the current page
    for idx in range(start_index, end_index):
        with st.expander(f"Question {idx + 1}"):
            item = dataset[idx]
            
            # Display question
            st.markdown("**Question:**")
            st.write(item.get('question', 'No question available'))
            
            # Display answer
            if st.button(f"Show Answer {idx + 1}"):
                st.markdown("**Answer:**")
                st.write(item.get('answer', 'No answer available'))

if __name__ == "__main__":
    view_dataset()
