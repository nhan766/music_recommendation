import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. CHUẨN BỊ DỮ LIỆU (Mock Data)
# Trong thực tế, bạn sẽ dùng pd.read_csv("dataset.csv")
@st.cache_data
def load_data():
    data = {
        'Song_ID': [1, 2, 3, 4, 5, 6],
        'Title': ['Shape of You', 'Bohemian Rhapsody', 'Perfect', 'Hotel California', 'Thinking Out Loud', 'Stairway to Heaven'],
        'Artist': ['Ed Sheeran', 'Queen', 'Ed Sheeran', 'Eagles', 'Ed Sheeran', 'Led Zeppelin'],
        'Genre': ['Pop', 'Rock', 'Pop', 'Rock', 'Pop', 'Rock']
    }
    return pd.DataFrame(data)

df = load_data()

# 2. HÀM TÍNH TOÁN GỢI Ý (Core Logic)
def recommend_songs(song_title, df, top_n=2):
    # Kết hợp đặc trưng (Genre và Artist) thành một chuỗi văn bản
    df['Features'] = df['Genre'] + " " + df['Artist']
    
    # Biểu diễn bài hát dưới dạng Vector (TF-IDF)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Features'])
    
    # Tính ma trận độ tương đồng Cosine (Cosine Similarity)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Lấy index của bài hát người dùng chọn
    idx = df[df['Title'] == song_title].index[0]
    
    # Lấy điểm số tương đồng của bài hát này với các bài khác
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sắp xếp các bài hát theo điểm số (từ cao xuống thấp)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Lấy top N bài hát (bỏ qua bài hát đầu tiên vì nó chính là bài người dùng chọn)
    sim_scores = sim_scores[1:top_n+1]
    song_indices = [i[0] for i in sim_scores]
    
    return df.iloc[song_indices]

# 3. GIAO DIỆN NGƯỜI DÙNG (Streamlit UI)
st.title("🎵 Music Recommendation System")
st.subheader("Team: NTH - IT Special Topic 1")

# Hiển thị bộ dữ liệu
st.write("### 📂 Dữ liệu bài hát hiện tại:")
st.dataframe(df)

# Khung chọn bài hát đầu vào
st.write("### 🎧 Chọn một bài hát bạn yêu thích:")
selected_song = st.selectbox("Bài hát:", df['Title'].values)

# Nút thực hiện gợi ý
if st.button("Gợi ý bài hát"):
    st.write(f"Đang tìm các bài hát tương tự như **{selected_song}**...")
    
    # Gọi hàm gợi ý
    recommendations = recommend_songs(selected_song, df, top_n=2)
    
    if not recommendations.empty:
        st.success("Tada! Dưới đây là các bài hát dành cho bạn:")
        # Hiển thị kết quả dưới dạng bảng đẹp mắt
        st.table(recommendations[['Title', 'Artist', 'Genre']])
    else:
        st.warning("Không tìm thấy bài hát phù hợp.")