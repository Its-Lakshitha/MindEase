<div align="center">

# 🧠 MindEase

### *Your Compassionate AI Mental Health Companion*

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org)
[![Django](https://img.shields.io/badge/Django-5.2.8+-green.svg)](https://www.djangoproject.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1+-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.57.1+-yellow.svg)](https://huggingface.co/transformers)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

**Talk • Reflect • Feel Better**

MindEase is an AI-powered conversational therapy bot designed to provide compassionate mental health support through evidence-informed conversations, guided reflection, and daily wellness check-ins.

[🚀 Get Started](#-quick-start) • [📚 Documentation](#-project-structure) • [🤝 Contributing](#-contributing) • [💬 Support](#-support)

</div>

---

## ✨ Features

<table>
<tr>
<td width="33%" align="center">
<h3>🔒 Confidential</h3>
<p>A safe, judgment-free space to explore your thoughts and feelings privately</p>
</td>
<td width="33%" align="center">
<h3>🧭 Guided Reflection</h3>
<p>Evidence-based prompts help you reframe situations and develop coping strategies</p>
</td>
<td width="33%" align="center">
<h3>📅 Daily Check-ins</h3>
<p>Build emotional resilience with personalized habit nudges and wellness tracking</p>
</td>
</tr>
<tr>
<td width="33%" align="center">
<h3>🎭 Emotion Recognition</h3>
<p>Advanced AI models detect and respond to emotional states in your messages</p>
</td>
<td width="33%" align="center">
<h3>🧠 Mental Health Classification</h3>
<p>Context-aware understanding of mental health conditions for appropriate support</p>
</td>
<td width="33%" align="center">
<h3>🌐 Multilingual Support</h3>
<p>Communicate in your preferred language with built-in translation capabilities</p>
</td>
</tr>
</table>

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Backend Framework** | Django 5.2.8+ |
| **AI/ML** | PyTorch 2.9.1+, Transformers 4.57.1+, Scikit-learn 1.7.2+ |
| **Data Processing** | Pandas 2.3.3+, HuggingFace Datasets 4.4.1+ |
| **Translation** | Deep-Translator 1.11.4+ |
| **Development** | Jupyter Notebooks, Python 3.12+ |
| **Database** | SQLite (development), PostgreSQL (production ready) |

---

## 🚀 Quick Start

### Prerequisites

- **Python** 3.12 or higher
- **Git** for version control
- **uv** or **pip** for package management

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/Its-Lakshitha/MindEase.git
cd MindEase
```

2. **Set up virtual environment**

```bash
# Using uv (recommended)
uv sync

# Or using traditional venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

3. **Download AI/ML Datasets** 📊

Navigate to the datasets directory:

```bash
mkdir -p models/datasets
cd models/datasets
```

Clone the required datasets from HuggingFace:

| Dataset | Purpose | Command |
|---------|---------|---------|
| 🎭 **Emotion Recognition** | Detect emotional states in user messages | `git clone https://huggingface.co/datasets/jobs-git/emotion-recognition-dataset` |
| 🧠 **Mental Health Classification** | Identify mental health conditions context | `git clone https://huggingface.co/datasets/sai1908/Mental_Health_Condition_Classification` |
| 💬 **Student Chat Data v2** | Training conversational responses | `git clone https://huggingface.co/datasets/chillies/student-mental-health-chat-data-v2` |
| 🗣️ **Student Conversations** | Dialogue pattern learning | `git clone https://huggingface.co/datasets/chillies/student-mental-health-conversation` |
| 🌏 **Counseling (Vietnamese)** | Multilingual support training | `git clone https://huggingface.co/datasets/chillies/student-mental-health-counseling-vn` |

4. **Initialize the database**

```bash
cd ../..  # Return to project root
python manage.py migrate
python manage.py createsuperuser  # Optional: for admin access
```

5. **Run the development server** 🚀

```bash
python manage.py runserver
```

Visit `http://localhost:8000` to see MindEase in action! 🎉

---

## 📂 Project Structure

```
MindEase/
┣━━ 🌐 main/                    # Main Django app
┃   ┣━━ templates/              # HTML templates
┃   ┣━━ views.py                # View controllers
┃   ┣━━ urls.py                 # URL routing
┃   ┗━━ models.py               # Database models
┃
┣━━ 🧠 MindEase/                # Django project settings
┃   ┣━━ settings.py             # Configuration
┃   ┣━━ urls.py                 # Root URL patterns
┃   ┗━━ wsgi.py                 # WSGI config
┃
┣━━ 🤖 models/                  # AI/ML models & data
┃   ┣━━ datasets/               # Training datasets (HuggingFace)
┃   ┣━━ preprocessing/          # Data preparation scripts
┃   ┣━━ processed/              # Cleaned & processed data
┃   ┗━━ src/                    # Model implementations
┃
┣━━ 📊 db.sqlite3               # Development database
┣━━ 📝 manage.py                # Django management script
┣━━ 📦 pyproject.toml           # Project dependencies
┗━━ 📖 README.md                # You are here!
```

---

## 🎯 Core Features in Detail

### 1. 🤖 AI-Powered Conversations
- Natural language understanding using transformer models
- Context-aware responses with emotional intelligence
- Adaptive conversation flow based on user sentiment

### 2. 🎭 Emotion Detection
- Real-time emotion recognition from text
- Multi-label classification (joy, sadness, anxiety, etc.)
- Emotion trend tracking over time

### 3. 🩺 Mental Health Support
- Evidence-based therapeutic techniques (CBT, DBT principles)
- Crisis detection and resource referral
- Personalized coping strategy recommendations

### 4. 📈 Progress Tracking
- Mood journaling and visualization
- Wellness habit streaks
- Personalized insights dashboard

---

## 🧪 Development

### Running Jupyter Notebooks

For model development and data exploration:

```bash
jupyter notebook
# Navigate to models/preprocessing/
```

### Training Models

```bash
cd models/src
python train_emotion_model.py
```

### Running Tests

```bash
python manage.py test
```

---

## 🗺️ Roadmap

- [x] 🎭 Emotion recognition system
- [x] 💬 Basic conversational interface
- [x] 🌐 Multi-language support
- [ ] 📱 Mobile app (React Native)
- [ ] 🔊 Voice-based interactions
- [ ] 📊 Advanced analytics dashboard
- [ ] 🤝 Group therapy sessions
- [ ] 🏥 Professional therapist integration

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 🔃 Open a Pull Request

Please read our [Contributing Guidelines](CONTRIBUTING.md) for more details.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 💬 Support

| Channel | Link |
|---------|------|
| 🐛 **Issues** | [GitHub Issues](https://github.com/Its-Lakshitha/MindEase/issues) |
| 💡 **Discussions** | [GitHub Discussions](https://github.com/Its-Lakshitha/MindEase/discussions) |
| 📧 **Email** | support@mindease.ai |
| 🌐 **Website** | [mindease.ai](https://mindease.ai) |

---

## ⚠️ Important Disclaimer

**MindEase is an AI assistant designed to support mental wellness, NOT a replacement for professional mental health care.**

🆘 **If you're experiencing a mental health crisis:**
- 🇺🇸 **USA**: Call 988 (Suicide & Crisis Lifeline)
- 🇬🇧 **UK**: Call 116 123 (Samaritans)
- 🌍 **International**: Visit [findahelpline.com](https://findahelpline.com)

---

## 🙏 Acknowledgments

- 🤗 **HuggingFace** for providing excellent datasets and transformer models
- 💚 **Django Community** for the robust web framework
- 🧠 **Mental Health Professionals** who informed our evidence-based approach
- 👥 **Open Source Contributors** who make projects like this possible

---

<div align="center">

**Made with ❤️ by the MindEase Team**

⭐ Star us on GitHub — it helps!

[🏠 Home](https://mindease.ai) • [📖 Docs](https://docs.mindease.ai) • [👥 Team](https://mindease.ai/team)

</div>
