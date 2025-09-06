import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import time
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Klasifikasi Kucing vs Anjing",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff6b6b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .sidebar-info {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def preprocess_image(image: Image.Image, target_size=(150, 150)):
    image = image.convert("RGB").resize(target_size)
    img_array = np.array(image).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = (img_array - mean) / std
    img_array = img_array.astype(np.float32)

    return np.expand_dims(img_array, axis=0)

@st.cache_resource
def load_model(onnx_path="cats_dogs.onnx"):
    providers = ['CPUExecutionProvider']
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.insert(0, 'CUDAExecutionProvider')
    return ort.InferenceSession(onnx_path, providers=providers)

def predict(session, img_array):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    start = time.time()
    pred = session.run([output_name], {input_name: img_array})[0]
    inference_time = time.time() - start

    prob = float(pred[0][0])
    label = "Dog" if prob > 0.5 else "Cat"
    confidence = prob if prob > 0.5 else 1 - prob
    return label, confidence, inference_time

def display_prediction_results(label, confidence, inference_time):
    st.markdown(f"""
    <div class="prediction-box">
        <h2>Prediksi Teratas</h2>
        <h1 style="color: #28a745; margin: 0;">{label}</h1>
        <h3 style="color: #6c757d; margin: 0;">Confidence: {confidence:.1%}</h3>
        <p style="margin-top: 1rem;">Inference time: {inference_time*1000:.1f} ms</p>
    </div>
    """, unsafe_allow_html=True)

    classes = ["Cat", "Dog"]
    probs = [1 - confidence, confidence] if label == "Dog" else [confidence, 1 - confidence]
    fig = px.bar(
        x=probs,
        y=classes,
        orientation="h",
        title="Confidence Scores (%)",
        color=probs,
        color_continuous_scale="Viridis"
    )
    fig.update_layout(showlegend=False, height=300, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    df = pd.DataFrame([
        {"Class": "Cat", "Confidence": f"{probs[0]:.1%}"},
        {"Class": "Dog", "Confidence": f"{probs[1]:.1%}"}
    ])
    with st.expander("Detail Prediksi"):
        st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown('<h1 class="main-header">Klasifikasi Kucing vs Anjing</h1>', unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>dengan model non-pretrained sederhana</h3>", unsafe_allow_html=True)

st.image(
        "https://img.freepik.com/premium-photo/dog-fighting-with-cat-white-background_161299-2175.jpg",
        caption="anjing atau kucing?",
        use_container_width=True
        )

with st.sidebar:
    session = load_model()
    st.markdown(f"""
    <div class="sidebar-info">
        <b>Model:</b> CNN Binary Classifier<br>
        <b>Kelas:</b> 2 (Cat, Dog)<br>
        <b>Ukuran input:</b> 150×150<br>
        <b>Runtime:</b> ONNX Runtime<br>
    </div>
    """, unsafe_allow_html=True)
    providers = session.get_providers()
    st.success(f"Model loaded ({'GPU' if 'CUDAExecutionProvider' in providers else 'CPU'})")
    show_original = st.checkbox("Tampilkan gambar asli", True)

st.markdown("## Upload gambar kucing atau anjing")
uploaded_file = st.file_uploader("Pilih gambar...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Input Image")
        if show_original:
            st.image(image, caption="Ukuran Asli", use_container_width=True)
        else:
            st.image(image.convert('RGB').resize((150, 150)), caption="Ukuran untuk di-input", use_container_width=True)
    with col2:
        if st.button("Mulai klasifikasi", type="primary"):
            img_array = preprocess_image(image)
            label, confidence, inference_time = predict(session, img_array)
            display_prediction_results(label, confidence, inference_time)

    with st.expander("Informasi Gambar"):
        st.write(f"**Format:** {image.format}")
        st.write(f"**Ukuran:** {image.size}")
        st.write(f"**Mode:** {image.mode}")
else:
    with st.expander("Deskripsi Dataset"):
        st.write("""
        Dataset Cats vs Dogs adalah dataset populer yang dirilis oleh Microsoft Research, yang dirancang khusus untuk tugas klasifikasi biner. Dataset ini berisi puluhan ribu gambar kucing dan anjing, menjadikannya salah satu dataset awal yang paling sering digunakan untuk melatih dan mengevaluasi model machine learning dan deep learning, terutama Convolutional Neural Networks (CNN).
        """)

    with st.expander("Format yang bisa di-upload"):
        st.write("""
        JPG/JPEG atau PNG
        """)

    with st.expander("Performa Model"):
        st.write("""
        Accuracy: 92.5%
        """)

    with st.expander("Deskripsi Model"):
        st.write("""
        Model ini memiliki tiga blok lapisan konvolusi, di mana setiap blok terdiri dari:

        1. nn.Conv2d (Lapisan Konvolusi): Mengekstrak fitur dari gambar.

            - Block 1: Mengubah 3 channel menjadi 32 channel.

            - Block 2: Mengubah 32 channel menjadi 64 channel.

            - Block 3: Mengubah 64 channel menjadi 128 channel.

        2. nn.BatchNorm2d (Normalisasi Batch): Menstabilkan pelatihan.

        3. nn.ReLU (Fungsi Aktivasi): Memperkenalkan non-linearitas.

        4. nn.MaxPool2d (Lapisan Pooling): Mengurangi dimensi gambar.

        Setelah mengekstrak fitur, model ini menggunakan lapisan-lapisan berikut untuk klasifikasi (FC Layer):

        1. nn.Flatten: Mengubah output dari lapisan konvolusi menjadi vektor 1D.

        2. nn.Dropout(0.5): Mencegah overfitting dengan menonaktifkan 50% neuron secara acak.

        3. nn.Linear (Lapisan Linear): Dua lapisan yang memetakan fitur ke output akhir.

            - Lapisan pertama: input dari 128 * 18 * 18, output 128.

            - Lapisan kedua: input 128, output 1 (probabilitas tunggal).

        4. nn.Sigmoid (Fungsi Aktivasi): Menghasilkan probabilitas akhir antara 0 dan 1.

        """)
