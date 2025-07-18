## Abstract
Graph contrastive learning (GCL) extracts informative representations from graphs with limited or no labels. For node classification, augmented views of a node are treated as positive pairs with high similarity. While current augmentation methods often disrupt semantic and structural properties, we propose a hop augmentation technique based on the homophily assumption. This method uses multi-hop neighborhood information to create multiple views from the network’s multi-head outputs, enhancing GCL by iteratively expanding node neighborhoods. It captures comprehensive graph information in sub-feature spaces without structural distortion, leveraging the insight that connected nodes share labels and features. Experiments on diverse benchmarks demonstrate superior node classification performance compared to state-of-the-art GCL methods, even with minimal annotations. 


## Citation
```
@article{zou2025multi,
  title={Multi-head graph contrastive learning with hop augmentation for node classification},
  author={Zou, Minhao and Wang, Yutong and Meng, Xiaofeng and Gan, Zhongxue and Guan, Chun and Leng, Siyang},
  journal={Pattern Recognition},
  pages={112055},
  year={2025},
  publisher={Elsevier}
}
```
