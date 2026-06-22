import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# CẤU HÌNH TRANG WEB
st.set_page_config(page_title="Music Recommendation System", page_icon="🎵", layout="wide")

st.markdown("""
    <style>
    /* Nền và phông chữ tổng thể */
    .main { background-color: #0e1117; }
    h1, h2, h3 { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    
    /* Phong cách thiết kế Thẻ bài hát (Music Card) */
    .music-card {
        background: linear-gradient(145deg, #1e222b, #15181f);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid #2d3139;
    }
    .music-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(29, 185, 84, 0.2);
        border-color: #1db954;
    }
    .music-title {
        color: #ffffff;
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 5px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .music-artist { color: #b3b3b3; font-size: 14px; margin-bottom: 10px; }
    .music-badge {
        background-color: #282c34;
        color: #1db954;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: bold;
        display: inline-block;
    }
    
    /* Thanh tiến trình độ tương đồng */
    .sim-bar-container {
        background-color: #282c34;
        border-radius: 10px;
        height: 6px;
        width: 100%;
        margin-top: 10px;
    }
    .sim-bar-fill {
        background: linear-gradient(90deg, #1db954, #1ed760);
        height: 6px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# 1. TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dataset.csv")
        df = df[['track_name', 'artists', 'track_genre']].dropna()
        df = df.drop_duplicates(subset=['track_name'])
        # Trích xuất 5000 bài hát ngẫu nhiên để tối ưu hóa hiệu năng ma trận
        return df.sample(n=5000, random_state=42).reset_index(drop=True)
    except:
        # Dữ liệu dự phòng trường hợp file CSV chưa sẵn sàng
        data = {
            'track_name': ['Shape of You', 'Bohemian Rhapsody', 'Perfect', 'Hotel California', 'Thinking Out Loud', 'Stairway to Heaven'],
            'artists': ['Ed Sheeran', 'Queen', 'Ed Sheeran', 'Eagles', 'Ed Sheeran', 'Led Zeppelin'],
            'track_genre': ['Pop', 'Rock', 'Pop', 'Rock', 'Pop', 'Rock']
        }
        return pd.DataFrame(data)

df = load_data()

# 2. HÀM TÍNH TOÁN THUẬT TOÁN (Trích xuất vector & Tính toán Cosine)
def recommend_songs(song_title, df, top_n=4):
    df['Features'] = df['track_genre'] + " " + df['artists']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    idx = df[df['track_name'] == song_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    
    results = []
    for i, score in sim_scores:
        row = df.iloc[i].copy()
        row['similarity_score'] = score
        results.append(row)
    return pd.DataFrame(results)

# 3. THIẾT KẾ GIAO DIỆN NGƯỜI DÙNG
st.markdown("<h1 style='text-align: center; color: #ffffff; margin-bottom: 5px;'>🎵 Smart Music Discovery</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #b3b3b3; margin-bottom: 30px;'>Hệ thống gợi ý bài hát thông minh dựa trên thuật toán Content-based Filtering</p>", unsafe_allow_html=True)

# Bố cục chia hai cột chính: Cột bên trái để chọn bài hát, cột bên phải hiển thị kết quả
# Bố cục chia hai cột chính: Cột bên trái để chọn bài hát, cột bên phải hiển thị kết quả
col_input, col_spacer, col_output = st.columns([1.5, 0.2, 3])

with col_input:
    st.markdown("<h3 style='color: #ffffff;'>🎧 Trình điều khiển</h3>", unsafe_allow_html=True)
    song_list = df['track_name'].tolist()
    
    # NÂNG CẤP: Biến Selectbox thành thanh tìm kiếm (Search Bar)
    selected_song = st.selectbox(
        "🔍 Nhập tên bài hát để tìm kiếm:", 
        options=song_list,
        index=None, # Để trống ban đầu
        placeholder="Gõ tên bài vào đây (VD: Shape of...)"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Chỉ cho phép bấm nút Tìm kiếm nếu đã chọn/gõ xong bài hát
    is_disabled = selected_song is None
    search_button = st.button("Phân tích & Gợi ý", use_container_width=True, disabled=is_disabled)
    
    if selected_song:
        # Hiển thị thông tin bài hát đang chọn hiện tại
        current_song_info = df[df['track_name'] == selected_song].iloc[0]
        st.markdown(f"""
            <div style='background-color: #14171c; padding: 15px; border-radius: 8px; border-left: 4px solid #1db954; margin-top: 30px;'>
                <div style='color: #b3b3b3; font-size: 12px; text-transform: uppercase;'>Đang chọn làm gốc</div>
                <div style='color: #ffffff; font-size: 16px; font-weight: bold; margin-top: 5px;'>{current_song_info['track_name']}</div>
                <div style='color: #888888; font-size: 14px;'>{current_song_info['artists']}</div>
            </div>
        """, unsafe_allow_html=True)

with col_output:
    if search_button and selected_song:
        st.markdown(f"<h3 style='color: #ffffff;'>✨ Danh sách đề xuất cho: {selected_song}</h3>", unsafe_allow_html=True)
        
        with st.spinner("Thuật toán đang tính toán khoảng cách vector..."):
            recommendations = recommend_songs(selected_song, df, top_n=4)
            
            grid_col1, grid_col2 = st.columns(2)
            for index, row in recommendations.iterrows():
                target_col = grid_col1 if index % 2 == 0 else grid_col2
                sim_percentage = int(row['similarity_score'] * 100)
                
                with target_col:
                    st.markdown(f"""
                        <div class="music-card">
                            <div class="music-title">🎵 {row['track_name']}</div>
                            <div class="music-artist">👤 {row['artists']}</div>
                            <span class="music-badge">🏷️ {row['track_genre'].upper()}</span>
                            <div style="color: #888; font-size: 11px; margin-top: 12px; display: flex; justify-content: space-between;">
                                <span>Độ tương đồng</span>
                                <span style="color: #1db954; font-weight: bold;">{sim_percentage}%</span>
                            </div>
                            <div class="sim-bar-container">
                                <div class="sim-bar-fill" style="width: {sim_percentage}%;"></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
    else:
        # Trạng thái mặc định
        st.markdown("<h3 style='color: #ffffff;'>Khám phá kho bài hát</h3>", unsafe_allow_html=True)
        st.info("Hãy gõ tên một bài hát vào ô tìm kiếm bên trái để hệ thống phân tích đặc trưng văn bản TF-IDF.")
        st.markdown("<p style='color: #b3b3b3; font-size: 14px;'>Gợi ý một số bài hát có trong bộ dữ liệu Kaggle:</p>", unsafe_allow_html=True)
        st.dataframe(df.head(6)[['track_name', 'artists', 'track_genre']].rename(columns={'track_name': 'Tên bài', 'artists': 'Nghệ sĩ', 'track_genre': 'Thể loại'}), use_container_width=True)
