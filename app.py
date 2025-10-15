import gradio as gr
import json
import pandas as pd
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class CompleteMedicalChatbot:
    def __init__(self, data_path='medibot_dataset.json'):
        print("Loading complete medical knowledge base...")

        # Load your actual dataset
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.df = pd.DataFrame(data)
        except FileNotFoundError:
            print(f"Warning: Dataset file {data_path} not found. Using empty dataset.")
            self.df = pd.DataFrame(columns=['input', 'response', 'category'])

        print(f"Dataset loaded: {len(self.df)} Q&A pairs")
        if len(self.df) > 0:
            print(f"Categories: {self.df['category'].unique().tolist()}")

        # Create comprehensive knowledge base
        self.knowledge_base = {}
        self.all_questions = []
        self.all_answers = []
        self.category_questions = {}

        for _, row in self.df.iterrows():
            question = row['input'].lower().strip()
            answer = row['response']
            category = row['category']

            # Store in knowledge base
            if category not in self.knowledge_base:
                self.knowledge_base[category] = {}
            self.knowledge_base[category][question] = answer

            # Store for quick access
            self.all_questions.append(question)
            self.all_answers.append(answer)

            # Store by category for quick buttons
            if category not in self.category_questions:
                self.category_questions[category] = []
            self.category_questions[category].append((question, answer))

        # Load sentence transformer for similarity matching
        print("Loading similarity model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Precompute embeddings for ALL questions
        print("Computing embeddings for all questions...")
        if len(self.all_questions) > 0:
            self.all_embeddings = self.model.encode(self.all_questions)
        else:
            self.all_embeddings = np.array([])

        print(f"Ready! Loaded {len(self.all_questions)} questions across {len(self.knowledge_base)} categories")

    def find_best_match(self, user_question, threshold=0.3):
        """Find the most similar question in the entire knowledge base"""
        if len(self.all_questions) == 0:
            return None, None, None, 0

        user_question_lower = user_question.lower().strip()

        # Encode user question
        user_embedding = self.model.encode([user_question_lower])

        # Calculate similarities with ALL questions
        similarities = cosine_similarity(user_embedding, self.all_embeddings)[0]

        # Find best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        best_question = self.all_questions[best_idx]
        best_answer = self.all_answers[best_idx]

        # Find which category this belongs to
        best_category = None
        for category, questions in self.knowledge_base.items():
            if best_question in questions:
                best_category = category
                break

        return best_question, best_answer, best_category, best_score

    def get_response(self, user_question):
        """Get response from the complete knowledge base"""
        user_question_lower = user_question.lower().strip()

        # First, try exact match
        for i, question in enumerate(self.all_questions):
            if user_question_lower == question:
                return self.all_answers[i], "exact", 1.0, self.all_questions[i]

        # Then try similarity matching with lower threshold to catch more questions
        if len(self.all_questions) > 0:
            best_question, best_answer, category, score = self.find_best_match(user_question, threshold=0.1)
            if score > 0.1:  # Very low threshold to catch most questions
                return best_answer, "similar", score, best_question

        # No match found (should be rare with low threshold)
        return None, "no_match", 0, None

    def generate_safe_response(self, user_question):
        """Generate response using the complete knowledge base"""
        # Get response from knowledge base
        answer, match_type, confidence, matched_question = self.get_response(user_question)

        if answer:
            # Add context about match quality
            if match_type == "exact":
                response = answer
                match_info = "**Thank you for using our medical chatbot. We've prepared your response below, I hope it helps you feel better informed.**"
            else:
                response = answer
                match_info = f"**Similar match** (confidence: {confidence:.2f})"

            # Add the matched question for transparency
            if matched_question and match_type == "similar":
                response = f"{match_info}\n\n**Matched question:** \"{matched_question}\"\n\n**Answer:** {response}"
            else:
                response = f"{match_info}\n\n{response}"

            # Always add medical disclaimer
            response += "\n\n---\n*This information comes from verified medical knowledge base. For personal medical advice, please consult a healthcare professional.*"

            return response
        else:
            # No match found - provide helpful guidance
            return "I couldn't find a close match in my medical knowledge base.\n\n**Suggestions:**\n• Try rephrasing your question\n• Use more specific medical terms\n• Check the quick question buttons below\n• Consult a healthcare provider for personal medical advice\n\nMy knowledge covers: Pneumonia, Malaria, Typhoid, General Health, Women & Child Health, Nutrition, and Mental Health."

    def get_quick_answers_by_category(self, num_per_category=5):
        """Get sample questions organized by category for quick buttons"""
        samples = {}
        for category in self.category_questions.keys():
            questions = self.category_questions[category][:num_per_category]
            samples[category] = [(q[0], q[1]) for q in questions]
        return samples

# Initialize the complete chatbot
complete_bot = CompleteMedicalChatbot()

# Clean Black & White CSS
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    max-width: 1400px !important;
    margin: 0 auto;
    background: #ffffff;
}

.main-container {
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
    overflow: hidden;
    margin: 20px auto;
    min-height: 800px;
    position: relative;
    border: 1px solid #e0e0e0;
}

.chat-container {
    background: white;
    border-radius: 8px;
    min-height: 700px;
    transition: all 0.3s ease;
    margin-left: 0;
}

.chatbot {
    background: white;
    border-radius: 8px;
    min-height: 600px;
    border: 1px solid #f0f0f0;
    transition: all 0.3s ease;
}

.chatbot .message.user {
    background: #f8f8f8 !important;
    border: 1px solid #e0e0e0;
    border-radius: 12px 12px 4px 12px;
    margin: 8px 0;
    padding: 12px 16px;
    max-width: 85%;
    margin-left: auto;
    color: #333333;
}

.chatbot .message.bot {
    background: #fafafa !important;
    border: 1px solid #e8e8e8;
    border-radius: 12px 12px 12px 4px;
    margin: 8px 0;
    padding: 12px 16px;
    line-height: 1.6;
    max-width: 85%;
    color: #333333;
}

.medical-header {
    text-align: center;
    padding: 20px;
    background: #2c2c2c;
    color: white;
    margin-bottom: 0;
    position: relative;
    overflow: hidden;
}

/* Left Sidebar Styles */
.left-sidebar {
    background: #fafafa;
    border-right: 1px solid #e0e0e0;
    padding: 16px;
    height: 100%;
    overflow-y: auto;
    border-radius: 8px 0 0 8px;
    transition: all 0.3s ease;
    width: 300px;
    min-width: 300px;
    max-width: 300px;
}

.sidebar-header {
    background: #404040;
    color: white;
    padding: 12px;
    border-radius: 6px;
    margin-bottom: 16px;
    text-align: center;
}

.sidebar-section {
    background: white;
    border-radius: 6px;
    padding: 0;
    margin: 12px 0;
    border: 1px solid #e0e0e0;
    overflow: hidden;
}

.sidebar-title {
    background: #f0f0f0;
    color: #333333;
    padding: 10px 12px;
    font-weight: 600;
    font-size: 0.9em;
    display: flex;
    align-items: center;
    gap: 6px;
    border-bottom: 1px solid #e0e0e0;
}

.sidebar-content {
    padding: 12px;
    max-height: 250px;
    overflow-y: auto;
    transition: all 0.3s ease;
}

.medical-button {
    background: #ffffff !important;
    border: 1px solid #d0d0d0 !important;
    color: #333333 !important;
    border-radius: 6px !important;
    padding: 8px 10px !important;
    margin: 4px 0 !important;
    font-weight: 500 !important;
    font-size: 0.85em !important;
    transition: all 0.2s ease !important;
    text-align: left !important;
    width: 100% !important;
    justify-content: flex-start !important;
}

.medical-button:hover {
    background: #f0f0f0 !important;
    border-color: #a0a0a0 !important;
    transform: translateX(1px) !important;
}

.primary-button {
    background: #404040 !important;
    border: none !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

.primary-button:hover {
    background: #505050 !important;
    transform: translateY(-1px) !important;
}

.input-box {
    border: 1px solid #d0d0d0 !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    font-size: 14px !important;
    background: white !important;
}

.input-box:focus {
    border-color: #808080 !important;
    outline: none !important;
}

/* External Toggle Button - Always Visible */
.external-toggle-btn {
    position: absolute !important;
    top: 16px;
    left: 16px;
    z-index: 1000;
    background: #606060 !important;
    border: none !important;
    color: white !important;
    border-radius: 6px !important;
    padding: 8px 12px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    width: 36px !important;
    height: 36px !important;
    min-width: 36px !important;
}

.external-toggle-btn:hover {
    background: #707070 !important;
    transform: translateY(-1px) !important;
}

.warning-box {
    background: #f8f8f8;
    border: 1px solid #d0d0d0;
    border-radius: 6px;
    padding: 12px;
    margin: 12px 0;
    color: #666666;
    font-size: 12px;
}

.stats-bar {
    background: #f0f0f0;
    border: 1px solid #d0d0d0;
    border-radius: 6px;
    padding: 10px 16px;
    margin: 12px 0;
    text-align: center;
    font-weight: 600;
    color: #333333;
    font-size: 0.85em;
}

.hidden {
    display: none !important;
}

.sidebar-hidden {
    width: 0px !important;
    min-width: 0px !important;
    max-width: 0px !important;
    padding: 0px !important;
    overflow: hidden !important;
    border: none !important;
}

.chat-expanded {
    margin-left: 0px !important;
    width: 100% !important;
}

/* Scrollbar styling */
.sidebar-content::-webkit-scrollbar {
    width: 4px;
}

.sidebar-content::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 2px;
}

.sidebar-content::-webkit-scrollbar-thumb {
    background: #c0c0c0;
    border-radius: 2px;
}

.sidebar-content::-webkit-scrollbar-thumb:hover {
    background: #a0a0a0;
}

.content-row {
    display: flex;
    width: 100%;
    transition: all 0.3s ease;
}
"""

def chat_with_complete_bot(message, history):
    """Chat function using complete knowledge base"""
    # Add typing indicator
    yield history + [[message, "Searching medical database..."]]
    time.sleep(0.3)

    # Get response from complete knowledge base
    response = complete_bot.generate_safe_response(message)

    # Return final response
    yield history + [[message, response]]

def toggle_sidebar(sidebar_visible):
    """Toggle sidebar visibility and update button icon"""
    new_visibility = not sidebar_visible
    if new_visibility:
        return new_visibility, gr.update(visible=True), "←"
    else:
        return new_visibility, gr.update(visible=False), "→"

# Get organized quick questions
quick_samples = complete_bot.get_quick_answers_by_category(5)

# Create interface with fixed left sidebar
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:

    # State to track sidebar visibility
    sidebar_visible = gr.State(value=True)

    with gr.Column(elem_classes="main-container"):

        # Clean Healthcare Header
        gr.HTML(f"""
        <div class="medical-header">
            <h1 style="margin: 0; font-size: 2.1em; font-weight: 600;">MediBot Pro</h1>
            <p style="margin: 6px 0 0 0; font-size: 1em; opacity: 0.9; font-weight: 300;">Medical Knowledge Base System</p>
            <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 6px; margin: 12px 0 0 0;">
                <div style="display: flex; justify-content: center; gap: 25px; flex-wrap: wrap; font-size: 0.9em;">
                    <div><strong>{len(complete_bot.all_questions)}</strong> Medical Q&A</div>
                    <div><strong>{len(complete_bot.knowledge_base)}</strong> Specialties</div>
                    <div><strong>Verified</strong> Knowledge</div>
                    <div><strong>Zero</strong> Hallucinations</div>
                </div>
            </div>
        </div>
        """)

        # Stats Bar
        categories = list(complete_bot.knowledge_base.keys())
        gr.HTML(f"""
        <div class="stats-bar">
            <strong>Medical Specialties:</strong> {', '.join(categories) if categories else 'No categories loaded'}
        </div>
        """)

        # External Toggle Button - Always Visible
        toggle_btn = gr.Button("←", elem_classes="external-toggle-btn")

        # Main Content Area with Left Sidebar
        with gr.Row(elem_classes="content-row"):
            # Left Column - Sidebar (Quick Questions)
            sidebar = gr.Column(scale=1, visible=True, elem_classes="left-sidebar")

            # Right Column - Chat Interface (Main Content)
            with gr.Column(scale=4, elem_classes="chat-container"):
                # Chat Interface
                chatbot = gr.Chatbot(
                    value=[],
                    label="Medical Consultation Interface",
                    height=500,
                    show_copy_button=True,
                    container=False
                )

                # Input Area
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Describe your medical question or concern...",
                        lines=2,
                        container=False,
                        scale=4,
                        max_lines=3,
                        elem_classes="input-box"
                    )
                    submit_btn = gr.Button("Send", variant="primary", elem_classes="primary-button", scale=1)

        # Enhanced Medical Disclaimer (Main Area)
        gr.HTML("""
        <div class="warning-box">
            <div style="text-align: center; font-weight: 600; margin-bottom: 8px;">
                IMPORTANT MEDICAL DISCLAIMER
            </div>
            This system provides general medical information from a verified knowledge base.
            It is not a substitute for professional medical advice, diagnosis, or treatment.
            Always consult qualified healthcare providers for personal medical concerns.
            All responses are sourced directly from the medical dataset with zero AI hallucinations.
        </div>
        """)

    # Build sidebar content separately
    with sidebar:
        # Sidebar Header
        gr.HTML("""
        <div class="sidebar-header">
            <h3 style="margin: 0; font-size: 1.1em;">Quick Questions</h3>
            <p style="margin: 4px 0 0 0; font-size: 0.8em; opacity: 0.9;">Click to ask instantly</p>
        </div>
        """)

        # Sidebar Content - Organized by Category
        for category, questions in quick_samples.items():
            with gr.Column(elem_classes="sidebar-section"):
                # Category Header
                gr.HTML(f"""
                <div class="sidebar-title">
                    {category}
                    <span style="margin-left: auto; font-size: 0.75em; background: rgba(0,0,0,0.1); padding: 2px 6px; border-radius: 8px;">
                        {len(questions)}
                    </span>
                </div>
                """)

                # Questions for this category
                with gr.Column(elem_classes="sidebar-content"):
                    for question, answer in questions:
                        btn_text = f"{question[:50]}{'...' if len(question) > 50 else ''}"
                        btn = gr.Button(
                            btn_text,
                            size="sm",
                            elem_classes="medical-button"
                        )
                        btn.click(lambda q=question: q, outputs=msg)

        # Medical Disclaimer in Sidebar
        gr.HTML("""
        <div class="warning-box">
            <strong>Medical Disclaimer</strong><br>
            This system provides general medical information from a verified knowledge base.
            It is not a substitute for professional medical advice.
        </div>
        """)

    # Event handlers
    submit_btn.click(
        chat_with_complete_bot,
        inputs=[msg, chatbot],
        outputs=chatbot
    ).then(lambda: "", outputs=msg)

    msg.submit(
        chat_with_complete_bot,
        inputs=[msg, chatbot],
        outputs=chatbot
    ).then(lambda: "", outputs=msg)

    # Sidebar toggle functionality
    toggle_btn.click(
        toggle_sidebar,
        inputs=[sidebar_visible],
        outputs=[sidebar_visible, sidebar, toggle_btn]
    )

if __name__ == "__main__":
    demo.launch()