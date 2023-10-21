import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = torch.load(r"C:\Users\OS\Desktop\speaker\weights\tmodel_62.pt", map_location=torch.device('cpu'))

print(model)
