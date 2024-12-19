from graphviz import Digraph
from PIL import Image as PILImage

def create_dynamic_hierarchical_graph(mcq_llm_items, oeq_llm_items, mcq_vlm_items, oeq_vlm_items, output_path):
    """
    Create a hierarchical graph dynamically based on four input lists and save it using Pillow.
    """
    # Create the graph object
    dot = Digraph(format='png', engine='dot')
    
    # Global attributes for orthogonal connections and styling
    dot.attr(rankdir='TB', splines='ortho')
    dot.attr('node', style='filled', fontname='Arial', fontsize='12', shape='box')
    
    # Root node with distinct style
    dot.node("SophoBench", "SophoBench", shape="ellipse", style="rounded,filled", fillcolor="lightblue", fontsize='14', fontweight="bold")
    
    # Add first level branches
    dot.node("LLM", "LLM", fillcolor="lightgreen")
    dot.node("VLM", "VLM", fillcolor="lightgreen")
    dot.edge("SophoBench", "LLM")
    dot.edge("SophoBench", "VLM")
    
    # LLM branch
    with dot.subgraph(name="cluster_LLM") as llm:
        llm.attr(label="LLM Branch", style="rounded", color="lightgrey")
        llm.node("MCQ_LLM", "L_MCQ", fillcolor="lightyellow")
        llm.node("OEQ_LLM", "L-OEQ", fillcolor="lightyellow")
        llm.edge("LLM", "MCQ_LLM")
        llm.edge("LLM", "OEQ_LLM")
        llm.node("MCQ_LLM_Box", "\n".join(mcq_llm_items), fillcolor="white")
        llm.node("OEQ_LLM_Box", "\n".join(oeq_llm_items), fillcolor="white")
        llm.edge("MCQ_LLM", "MCQ_LLM_Box")
        llm.edge("OEQ_LLM", "OEQ_LLM_Box")
    
    # VLM branch
    with dot.subgraph(name="cluster_VLM") as vlm:
        vlm.attr(label="VLM Branch", style="rounded", color="lightgrey")
        vlm.node("MCQ_VLM", "V-MCQ", fillcolor="lightyellow")
        vlm.node("OEQ_VLM", "V-OEQ", fillcolor="lightyellow")
        vlm.edge("VLM", "MCQ_VLM")
        vlm.edge("VLM", "OEQ_VLM")
        vlm.node("MCQ_VLM_Box", "\n".join(mcq_vlm_items), fillcolor="white")
        vlm.node("OEQ_VLM_Box", "\n".join(oeq_vlm_items), fillcolor="white")
        vlm.edge("MCQ_VLM", "MCQ_VLM_Box")
        vlm.edge("OEQ_VLM", "OEQ_VLM_Box")
    
    # Render the graph to a temporary PNG file
    temp_path = dot.render(filename="sopho", cleanup=True)
    
    # Open the image using Pillow and save to the specified path
    image = PILImage.open(temp_path)
    image.save(output_path)

# Example usage
mcq_llm_items = ["Ai2Arc", "BigBenchHard(BBH)", "Medical Meadow MedQA", "MedMCQA", "MedQA", "MedQA USMLE 4Options", "MMLU", "MMLU Pro", "SciQ", "WinoGrande"]
oeq_llm_items = ["GPQA", "GSM8K", "GSMPlus", "IMO Geometry", "MathQA", "Medical Meadow FlashCards", "Medical Meadow Wikidoc Patient", "Medical Questions", "MedQna Version3", "MedQuAd", "MetaMathQA", "MetaMathQA40K", "SciBench", "SimpleQA", "TrugthfuklQA"]
mcq_vlm_items = ["Ai2D", "BLINK", "Cauldron", "MathV360K", "MMMU", "NEJM", "ScienceQA", "WorldMedQA"]
oeq_vlm_items = ["Animals", "Camou", "Captcha", "KvasirQA", "MathVision", "MathVista", "MedTrinity25M", "OlympiadBench","OlympicAreana", "PD12M", "RealWorldQA", "RocoRadiology", "SLAKE", "TheoremQA", "VisitBench", "VlmsareBlind", "VQRad" ]

# Save the graph
output_image_path = "sopho.png"
create_dynamic_hierarchical_graph(mcq_llm_items, oeq_llm_items, mcq_vlm_items, oeq_vlm_items, output_image_path)

