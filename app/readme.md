# Brain Tumor MRI Classifier (Streamlit App)

This is a **Streamlit web app** for brain tumor classification from MRI images using two models:

- **Pre-trained PyTorch CNN** with Grad-CAM visualization  
- **Custom CNN Framework** with layer-wise feature map inspection

Supports 4 classes:  
**Normal**, **Meningioma**, **Glioma**, **Pituitary**

---

## Features

### PyTorch Model
- Grad-CAM visualization for single image
- Batch prediction with result export to Excel
- Bar chart of class probabilities

### Custom CNN Model
- Layer-by-layer feature map visualization
- Support for both single and batch prediction
- Feature maps saved as `.png` or `.npy`

---

## Requirements

Install dependencies:

```bash
pip install streamlit torch torchvision opencv-python matplotlib pandas scikit-learn pytorch-grad-cam
```
## Run app
1. **Main display**
![Dashboard](./images/giao_dien.png)  
*Figure 1: Dashboard*
2. **Chosing image to test**
![Single image](./images/chon_anh_don.png)  
*Figure 2: Single image*
3. **Check prediction**
![Check prediction](./images/kiem_tra_du_doan.png)  
*Figure 3: Check prediction*
4. **Check prediction**
![Check grad-cam](./images/kiem_tra_grad_cam.png)  
*Figure 4: Check grad-cam*
5. **Check prediction**
![Check feature-map](./images/kiem_tra_feature_map.png)  
*Figure 5: Check feature-map*
6. **Chosing multi images**
![Chosing multi images](./images/chon_nhieu_anh.png)  
*Figure 6: Chosing multi images*
7. **Check multi result**
![Check multi result](./images/ket_qua_du_doan_nhieu_anh.png)  
*Figure 7: Check multi result*