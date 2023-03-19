import os

import torch
import numpy as np
from PIL import Image
import streamlit as st


from model import SimpleCNN
from torchvision import transforms

class StreamlitApp:
    def __init__(self, image_path, model_path):
        self.image_path = image_path
        self.model_path = model_path

    def __load_model(self):
        model = SimpleCNN()
        model.load_state_dict(torch.load(self.model_path))
        return model
    
    def __get_processsed_image(self, image):
        transform = transforms.Compose([   
                transforms.ToTensor(), # convert image numpy array to tensor
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # normalise the tensor (mean, stdev)
                transforms.Resize([32,32])
                ])
        transformed_image = torch.unsqueeze(transform(image), dim=0)
        return transformed_image
    
    def __get_label(self, label_id):
        label_dict = {
            "Airplane" : 0,
            "Automobile" : 1,
            "Bird" : 2,
            "Cat" : 3,
            "Deer" : 4,
            "Dog" : 5,
            "Frog" : 6,
            "Horse" : 7,
            "Ship" : 8,
            "Truck" : 9,
        }

        rev_label_dict = {j:i for i, j in label_dict.items()}
        return rev_label_dict[label_id[0]]


    def load_image(self):
        if self.image_path is not None:
            image = Image.open(self.image_path)
            st.image(image, caption="Uploaded Image")
        return np.array(image)

    def get_output_label(self, image):
        processed_image = self.__get_processsed_image(image=image)
        model = self.__load_model()
        output = model(processed_image)
        label_id = torch.argmax(output, dim=1).detach().cpu().numpy()
        label = self.__get_label(label_id=label_id)
        st.text_area(label="Predition", value=label)
        
    

if __name__ == "__main__":
    st.set_page_config(layout='wide')
    c1, c2 = st.columns(spec=2)

    with c1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        print(type(uploaded_file))  # Add this line to print the type of uploaded_file

        if uploaded_file is not None:
            app = StreamlitApp(image_path=uploaded_file, model_path="model.pt")
            image = app.load_image()
        else:
            st.write("No image uploaded.")

    with c2:
        if st.button("Predict"):
            label = app.get_output_label(image=image)
