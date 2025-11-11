import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from .data_loader import load_datas


def training(model, train_loader, optimizer, device):
	model.train()
	correct_count, loss_epoch = 0, 0
	for img, label in train_loader:
		img, label = img.to(device), label.to(device)
		optimizer.zero_grad()
		output = model(img) # inference with model
		pred = output.argmax(dim=1, keepdim=True)
		correct_count += pred.eq(label.view_as(pred)).sum().item()

		loss = F.cross_entropy(output, label.squeeze())
		loss_epoch += loss.item()
		loss.backward()
		optimizer.step()
	loss_epoch /= len(train_loader) # number of batches in one epoch
	correct_rate = correct_count / len(train_loader.dataset) # number of samples in all dataset
	return loss_epoch, correct_rate


def validation(model, val_loader, device):
	model.eval()
	correct_count, loss_val = 0, 0
	with torch.no_grad():
		for img, label in val_loader:
			img, label = img.to(device), label.to(device)
			output = model(img)
			pred = output.argmax(dim=1, keepdim=True)
			correct_count += pred.eq(label.view_as(pred)).sum().item()
	loss_val /= len(val_loader)
	correct_rate = correct_count / len(val_loader.dataset)
	return loss_val, correct_rate


def test(model, test_loader, device):
	print("[Predicting on test data...]")
	prediction = []
	model.eval()
	with torch.no_grad():
		for (img, ) in test_loader:
			img = img.to(device)
			output = model(img)
			pred = output.argmax(dim=1, keepdim=True)
			prediction.append(pred.cpu().numpy())
	prediction_arr = np.array([p.item() for p in prediction]).reshape(-1, 1)
	np.savetxt("prediction.csv", prediction_arr, fmt="%d", delimiter=";")
	print("[Predcitons are saved => (predictions.csv)]")
	return prediction_arr
	

def train_model(model, datas, device, lr=0.00005):
	X_train, X_val, y_train, y_val = datas
	train_loader, val_loader = load_datas(X_train, X_val, y_train, y_val)
	optimizer = optim.Adam(model.parameters(), lr=0.00005)
	loss_epoch = None
	counter = 0
	print("[Training started]")

	# loss_train = []
	# loss_val = []
	# acc_train = []
	# acc_val = []
	records = [[], [], [], []]
	for epoch in range(40):
		new_val_loss, acc_val = validation(model, val_loader, device)
		new_train_loss, acc_train = training(model, train_loader, optimizer, device)

		if loss_epoch is not None and abs(new_train_loss - loss_epoch) < 0.001:
			counter += 1
		else:
			counter = 0
		if counter >= 3:
			print("[Early stopped.]")
			break

		record = [new_train_loss, new_val_loss, acc_train, acc_val]
		print(f"[Epoch]: {epoch}  [Train Loss]: {record[0]:.4f}  [Train Acc]: ({(record[2] * 100):.0f}%)  [Val Loss]: {record[1]:.4f}  [Val Acc]: ({(record[3] * 100):.0f}%)")
		append_records(records, record)
		loss_epoch = new_train_loss
		print("[Training done.]")
		save_model(model, "weights.pth")
		show_records(records)
		return model


def append_records(records, record):
    new_loss_epoch, new_val_loss, acc_train, acc_val = record
    loss_train_all, loss_val_all, acc_train_all, acc_val_all = records

    loss_train_all.append(float(new_loss_epoch))
    loss_val_all.append(float(new_val_loss))
    acc_train_all.append(float(acc_train))
    acc_val_all.append(float(acc_val))


def show_records(records):
    loss_train_all, loss_val_all, acc_train_all, acc_val_all = records
    plt.title("Loss curves")
    plt.plot(loss_train_all, label="Train loss")
    plt.plot(loss_val_all, label="Validation loss")
    plt.legend(loc="upper right")
    plt.grid(linestyle="--", alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Accuracy curves")
    plt.plot(acc_train_all, label="Train accuracy")
    plt.plot(acc_val_all, label="Validation accuracy")
    plt.legend(loc="lower right")
    plt.grid(linestyle="--", alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()


def use_device():
    print(f"[CUDA => ({torch.cuda.is_available()})]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"[Model saved => ({path})]")