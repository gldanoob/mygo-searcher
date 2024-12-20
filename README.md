# MyGO 梗圖語義搜尋器

協助賣糕人正常對話的工具
- **語義搜尋**不用配對關鍵詞字符的方式，而是根據關鍵詞的意思作搜索，支援不同用詞，甚至跨語言的搜索
    - 例如：搜尋「so what」就可以找到愛音「那又怎樣？」的圖

## 使用方式
- 建立，啓動虛擬環境
`python3 -m venv venv`
`source venv/bin/activate`

- 安裝依賴
`pip install -r requirements.txt`

## 原理
在資料集建立過程中，該工具使用一個預訓練的文本嵌入模型，將標註集合轉換為歐幾里得空間中的向量。當提交查詢時，該查詢被轉換為一個向量，並使用餘弦相似度來找到資料集中與其最相似的k個條目。

具體地說，對於嵌入在 $\mathbb{R}^n$ 中的向量，兩個向量 $\vec{a}$ 和 $\vec{b}$ 的餘弦相似度被定義為
$$\frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\| \|\vec{b}\|}\text{。}$$。

由於在歐幾里得空間中點積的對偶性，餘弦相似度等價於向量 $\vec{a}$ 和 $\vec{b}$ 之間角度的餘弦。句子嵌入的一個性質是嵌入空間的「語義方向」，這可由著名的「國王 - 男人 + 女人 = 女王」類比來說明。餘弦相似度利用此性質來檢查查詢與資料集條目之間的角度。

搜尋的優化是透過將資料集打包到一個特徵矩陣 $\boldsymbol{X}$ 中並計算 $\boldsymbol{X} \cdot \vec{q}$ 來完成，其中 $\vec{q}$ 是查詢向量。然後將這些條目標準化並按餘弦相似度排序。

## 自訂數據集