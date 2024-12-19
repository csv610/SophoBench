import streamlit as st
from ai2arc import Ai2ArcDataset
from mmlu import MMLUDataset
from mmlu_pro import MMLUProDataset
from medmcqa import MedMcqaDataset
from medqa import MedQADataset
from medqa_usmle_4_options import MedQAUSMLE4OptionsDataset
from sciq import SciQDataset
from winogrande import WinoGrandeDataset
from bigbenchhard import BigBenchHardDataset
from medical_meadow_medqa import MedicalMeadowMedQADataset

st.set_page_config(layout="wide")

# Map of dataset names to their handler classes
DATASET_HANDLERS = {
    "AI2 ARC": Ai2ArcDataset,
    "MMLU": MMLUDataset,
    "MMLU Pro": MMLUProDataset,
    "MedMCQA": MedMcqaDataset,
    "MedQA": MedQADataset,
    "MedQA USMLE 4 Options": MedQAUSMLE4OptionsDataset,
    "SciQ": SciQDataset,
    "Winogrande": WinoGrandeDataset,
    "BigBench Hard": BigBenchHardDataset,
    "Medical Meadow MedQA": MedicalMeadowMedQADataset
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
        selected_page = st.sidebar.selectbox("Select Page Number", options=page_options)

        # Display items for the selected page
        start_index = (selected_page - 1) * num_items_per_page
        end_index = min(start_index + num_items_per_page, total_items)

        return dataset_handler, dataset, start_index, end_index
    else:
        st.sidebar.error("No data available for selected split")
        return None, None, 0, 0

def view_dataset():
    """
    Display the dataset in a paginated format with questions and choices.
    Handles the main UI layout and interaction logic.
    """
    st.title("Dataset Viewer")
    st.divider()

    # Get sidebar selections
    dataset_handler, dataset, start_index, end_index = config_panel()
    
    if not dataset or not dataset_handler:
        st.error("Failed to load dataset. Please check your selections.")
        return

    for idx in range(start_index, end_index):
        row = dataset[idx]
        question_id = f"q_{idx}"
        st.header(f"Question: {idx + 1}")

        data = dataset_handler.extract_data(row)
        
        # Display question
        question = data.get('question', '')
        if not question:
            st.warning("Question text not available")
            continue
            
        st.write(question)

        options = data.get('options', [])
        if not options:
            st.warning("No options available for this question")
            continue

        # Display choices
        for i, option in enumerate(options):
            st.write(f"({chr(65+i)}) {option}")

        # Create two columns for answers
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(f"Show Correct Answer", key=f"correct_{question_id}"):
                correct_answer = dataset_handler.get_correct_answer(row)
                if correct_answer:
                    st.success(f"Correct Answer: ({correct_answer})")
                else:
                    st.warning("Correct answer not available")
        
        with col2:
            if st.button(f"Show Model Answer", key=f"model_{question_id}"):
                response = dataset_handler.process_question(row)
                if response:
                    st.info(f"Model Response: {response}")
                else:
                    st.warning("Model response not available")

        st.divider()

if __name__ == "__main__":
    view_dataset()
