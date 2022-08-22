from sentence_transformers import SentenceTransformer, models, InputExample, losses
from torch import nn
from torch.utils.data import DataLoader
import random

def bert_model():
	word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
	pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
	dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())
	return SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

def get_data(in_path="./data/couplet/train/in.txt", out_path="./data/couplet/train/out.txt", vocab_path="./data/vocab.txt"):
	f = open(in_path, encoding="utf-8")
	datain = []
	for data in f.read().split("\n"):
		datain.append(data)

	f = open(out_path, encoding="utf-8")
	dataout = []
	for data in f.read().split("\n"):
		dataout.append(data)

	data = []
	for i in range(len(datain)):
		data.append(InputExample(texts=[datain[i], dataout[i]], label=1.0))
	print(len(data))

	f = open(vocab_path, encoding="utf-8")
	vocab = f.read()
	f.close()
	for i in range(1000000 - len(data)):
		datain = random.choice(datain)
		dataout = ""
		for j in range(len(datain)):
			dataout += random.choice(vocab)
		data.append(InputExample(texts=[datain, dataout], label=0.0))

	return data

if __name__ == '__main__':
	model = bert_model()
	train_dataloader = DataLoader(get_data(), shuffle=True, batch_size=400)
	train_loss = losses.CosineSimilarityLoss(model)
	model.fit(train_objectives=[(train_dataloader, train_loss)], device="cuda:1", epochs=5, warmup_steps=100, output_path="./model/STModel.dat")
