# Implementation of 'Semi-Supervised Learning via Compact Latent Space Clustering'

---
This is a TensorFlow2 implementation of the method described in the paper 'Semi-Supervised Learning via Compact Latent Space Clustering' by Konstantinos Kamnitsas, Daniel C. Castro, Loic Le Folgoc, Ian Walker, Ryutaro Tanno, Daniel Rueckert, Ben Glocker, Antonio Criminisi, Aditya Nori.

**Paper Link:** https://arxiv.org/pdf/1806.02679

**Data:** 利用 MNIST 手寫數字數據集實現本篇論文。

**Dependencies**
```
Python: 3.7
TensorFlow: 2.6.2
```

**Method:**
- 此篇論文提出一種用於半監督式學習的新型成本函數，以促進潛在空間的緊密群聚，從而提高分離效果。該方法涉及在帶有標記和未標記樣本的特徵空間上創建 Graph，以捕獲特徵空間中的潛在結構。
- 正則化技術: 其所提出的損失函數基於 Graph 上的 Markov Chains ，用於調節 latent space ，鼓勵每個類別形成一個單一緊密的集群。
- Skill: SSL、Graph、Label Propagation、Markov Chains

**load_data.py:**
- 所有類別各隨機挑選10筆資料用作監督式學習的 labeled data ，共100筆。
- 剰餘資料用於非監督式學習，每個批次在所有類別中各隨機挑選10筆資料用作 un-labeled data。
- 為了方便視覺化觀察，在此依照數字順序做排序。

**network.py:** 參考原論文，並稍微調整一些參數。

**main.py:** 透過 cclp loss 和 id loss 進行模型參數更新。


---

### Acknowledgments
The code is basically a modification of [ssl_compact_clustering](https://github.com/Kamnitsask/ssl_compact_clustering) implemented in TensorFlow 2. All credit goes to the authors of '[Semi-Supervised Learning via Compact Latent Space Clustering](https://arxiv.org/abs/1806.02679)', Konstantinos Kamnitsas, Daniel C. Castro, Loic Le Folgoc, Ian Walker, Ryutaro Tanno, Daniel Rueckert, Ben Glocker, Antonio Criminisi, Aditya Nori.

[[Paper]](https://arxiv.org/abs/1806.02679) 
[[Code(TensorFlow1)]](https://github.com/Kamnitsask/ssl_compact_clustering)
```
@inproceedings{Kamnitsas2018SemiSupervisedLV,
  title={Semi-Supervised Learning via Compact Latent Space Clustering},
  author={Konstantinos Kamnitsas and Daniel C. Castro and Lo{\"i}c Le Folgoc and Ian Walker and Ryutaro Tanno and Daniel Rueckert and Ben Glocker and Antonio Criminisi and Aditya V. Nori},
  booktitle={International Conference on Machine Learning},
  year={2018}
}
```


---
### References
- https://github.com/Kamnitsask/ssl_compact_clustering

