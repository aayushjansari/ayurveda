# 🌿 Ayurveda Knowledge Bot

An AI-powered chatbot for traditional Ayurvedic medicine consultation, built with Streamlit, ChromaDB, and LangChain.

## ✨ Features

- 📄 **PDF Document Processing** - Intelligent chunking and embedding of Ayurvedic texts
- 🔍 **Vector Similarity Search** - ChromaDB-powered semantic search
- 🤖 **RAG System** - LangChain-based retrieval-augmented generation
- 💬 **Interactive Chat Interface** - Modern Streamlit-based UI
- 🌐 **Cloud Ready** - Optimized for Streamlit Cloud deployment

## 🚀 Live Demo

**Try it now:** [Your App URL will be here after deployment]

## 🛠️ Local Development

### Prerequisites
- Python 3.8+
- Git

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/ayurveda-bot.git
   cd ayurveda-bot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add PDF documents**
   - Place your Ayurvedic PDF documents in the `knowledge_base_files/` folder

4. **Run the application**
   ```bash
   streamlit run chat_bot.py
   ```

5. **Initialize the system**
   - Use the sidebar to initialize the document processing and vector database
   - Start chatting with your Ayurvedic knowledge base!

## 📋 Configuration

### Environment Variables (Optional)
- `OPENAI_API_KEY` - For OpenAI embeddings and LLM (fallback to HuggingFace if not provided)
- `GOOGLE_API_KEY` - For Google Gemini LLM support

### Local vs Cloud
The app automatically detects the deployment environment and adjusts:
- **Local**: Persistent storage with saved state
- **Cloud**: Fresh processing on each restart (perfect for demos)

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Documents │───▶│  Document        │───▶│   ChromaDB      │
│                 │    │  Processing      │    │   Vector Store  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Streamlit UI   │◄───│  LangChain       │◄───│  Vector Search  │
│                 │    │  RAG Chain       │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔧 Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python, FastAPI-compatible design
- **Vector Database**: ChromaDB
- **LLM Framework**: LangChain
- **Embeddings**: HuggingFace Sentence Transformers / OpenAI
- **PDF Processing**: PyPDF, LangChain document loaders
- **Deployment**: Streamlit Cloud, Railway, Render

## 📝 Usage Examples

### Sample Questions to Try:
- "What are the three doshas in Ayurveda?"
- "How to balance Vata dosha?"
- "Ayurvedic remedies for digestion problems"
- "What is Panchakarma treatment?"
- "Benefits of turmeric in Ayurveda"

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This application is for educational and informational purposes only. The information provided should not be considered as medical advice. Always consult with qualified Ayurvedic practitioners or healthcare professionals for personalized medical guidance.

## 🙏 Acknowledgments

- Built using modern LangChain patterns from Context7 documentation
- Inspired by traditional Ayurvedic knowledge systems
- Powered by open-source AI and vector database technologies

---

**Built with ❤️ for the Ayurvedic community**
