import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchinfo import summary
import math
from prettytable import PrettyTable
from dataclasses import dataclass
import time

def print_params(module, print_vals=True):
	params_table = PrettyTable(['module', 'num_params', 'requires_grad'])
	total_trainable_params = 0
	for name, param in module.named_parameters():
		params_table.add_row([name, param.numel(), param.requires_grad])
		if param.requires_grad:
			total_trainable_params = total_trainable_params + param.numel()
	print(params_table)
	if total_trainable_params > 1e6:
		print(f'total trainable params: {(total_trainable_params / 1e6):0.2f}M')
	else:
		print(f'total trainable params: {total_trainable_params}')

@dataclass
class TestTogepiConfig:
	# embedding
	vocab_size = 10  # includes special tokens ([PAD], [MASK], [CLS], [SEP]) 
	padding_idx = 0
	max_position_embeddings = 7  # includes proxy for padding token; max_length = max_position_embeddings - 1
	pad_position = 0
	num_token_types = 3  # includes padding token type
	pad_token_type = 0
	embedding_dim = 4
	embedding_dropout_proba = 0.1
	
	# attention
	causal_attn = True  # for generative pre-training
	num_attn_heads = 2
	attn_actn = 'gelu'
	sparse_dens = 0.3
	attn_dropout_proba = 0.1

#test_config = TestTogepiConfig()
#test_config.vocab_size


class Embedding(nn.Module):
	def __init__(self, config):
		super().__init__()

		self._padding_idx = config.padding_idx
		self._pad_position = config.pad_position
		self._pad_token_type = config.pad_token_type

		self.tok_emb = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_dim, padding_idx=config.padding_idx)
		self.pos_emb = nn.Embedding(num_embeddings=config.max_position_embeddings, embedding_dim=config.embedding_dim, padding_idx=config.pad_position)
		self.type_emb = nn.Embedding(num_embeddings=config.num_token_types, embedding_dim=config.embedding_dim, padding_idx=config.pad_token_type)

		nn.init.xavier_uniform_(self.tok_emb.weight.data)
		self.tok_emb.weight.data[self._padding_idx] = torch.zeros(config.embedding_dim)
		nn.init.xavier_uniform_(self.pos_emb.weight.data)
		self.tok_emb.weight.data[self._pad_position] = torch.zeros(config.embedding_dim)
		nn.init.xavier_uniform_(self.type_emb.weight.data)
		self.tok_emb.weight.data[self._pad_token_type] = torch.zeros(config.embedding_dim)

		self.layer_norm = nn.LayerNorm(normalized_shape=config.embedding_dim, eps=1e-12)
		self.dropout = nn.Dropout(p=config.embedding_dropout_proba)

	def forward(self, input_ids, token_type_ids=None, padding_mask=None):
		# input_ids: (batch_size, max_length)
		# padding_mask: (batch_size, max_length)
		max_length = input_ids.shape[1]
		# assert(max_length == self.pos_emb.num_embeddings - 1)
		if padding_mask is None:
			# 1: no pad, 0: pad
			padding_mask = torch.where(input_ids == self._padding_idx, 0, 1)

		# position_ids: (batch_size, max_length)
		# assert(self._pad_position == 0)
		position_ids = torch.arange(max_length, dtype=torch.long, device=input_ids.device) + 1  # assuming zero is reserved for pad position
		position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
		position_ids = position_ids.masked_fill(padding_mask == 0, self._pad_position)

		# token_type_ids: (batch_size, max_length)
		if token_type_ids is None:
			# assert(self._pad_token_type == 0)
			token_type_ids = torch.ones_like(input_ids)  # assuming zero is reserved for pad position
		token_type_ids = token_type_ids.masked_fill(padding_mask == 0, self._pad_token_type)
		
		token_embeddings = self.tok_emb(input_ids)
		position_embeddings = self.pos_emb(position_ids)
		token_type_embeddings = self.type_emb(token_type_ids)
		
		return self.dropout(self.layer_norm(token_embeddings + position_embeddings + token_type_embeddings))

#test_input_ids = torch.tensor([[1, 2, 3, 4, 0, 0], [3, 4, 5, 6, 7, 8]])
#test_emb_obj = Embedding(test_config)
#test_emb = test_emb_obj(test_input_ids)
#print_params(test_emb_obj)
#test_emb, test_emb.shape


class MultiHeadAttention(nn.Module):
	def __init__(self, config):
		super().__init__()

		assert(config.embedding_dim % config.num_attn_heads == 0)

		self._num_heads = config.num_attn_heads
		self._per_head_dim = config.embedding_dim // config.num_attn_heads
		max_length = config.max_position_embeddings - 1

		self.wq = nn.Linear(in_features=config.embedding_dim, out_features=config.embedding_dim)
		self.wk = nn.Linear(in_features=config.embedding_dim, out_features=config.embedding_dim)
		self.wv = nn.Linear(in_features=config.embedding_dim, out_features=config.embedding_dim)
		nn.init.xavier_normal_(self.wq.weight.data)
		nn.init.xavier_normal_(self.wk.weight.data)
		nn.init.xavier_normal_(self.wv.weight.data)

		self._causal = config.causal_attn
		if config.causal_attn:
			self.register_buffer('causal_attn_mask', torch.tril(torch.ones(max_length, max_length)).view(1, 1, max_length, max_length))

		self.wo = nn.Linear(in_features=config.embedding_dim, out_features=config.embedding_dim)
		nn.init.xavier_normal_(self.wo.weight.data)

		self.layer_norm = nn.LayerNorm(normalized_shape=config.embedding_dim, eps=1e-12)
		self.dropout = nn.Dropout(p=config.attn_dropout_proba)
		self.softmax = nn.Softmax(dim=-1)
	
	def _extend_padding_mask(self, padding_mask, embeddings):
		# padding_mask: (batch_size, max_length)
		if padding_mask is None:
			padding_mask = torch.ones(embeddings.shape[0], embeddings.shape[1])

		extended_padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
		extended_padding_mask = extended_padding_mask.to(dtype=embeddings.dtype)  # amp/fp16 compatibility
		extended_padding_mask = (1 - extended_padding_mask) * -1e4
		return extended_padding_mask

	def forward(self, embeddings, padding_mask=None):
		batch_size = embeddings.shape[0]
		max_length = embeddings.shape[1]
		embedding_dim = embeddings.shape[2]

		# embeddings: (batch_size, max_length, embedding_dim)
		# attn_mask: 1 = non-pad, 0 = pad
		# projected_*: (batch_size, max_length, num_heads * per_head_dim)
		projected_query = self.wq(embeddings)
		projected_key = self.wk(embeddings)
		projected_value = self.wv(embeddings)

		sliced_projected_query = projected_query.view(batch_size, max_length, self._num_heads, self._per_head_dim).permute(0, 2, 1, 3)
		sliced_projected_key_tr = projected_query.view(batch_size, max_length, self._num_heads, self._per_head_dim).permute(0, 2, 3, 1)
		sliced_projected_value = projected_query.view(batch_size, max_length, self._num_heads, self._per_head_dim).permute(0, 2, 1, 3)

		# attn_mat: (batch_size, num_heads, max_length, max_length)
		# attn_mat: QK' / sqrt(d)
		# attn_mask: set [pad] tok attn values to -inf
		attn_mat = torch.matmul(sliced_projected_query, sliced_projected_key_tr) / np.power(embedding_dim, 0.5)
		attn_mat = attn_mat + self._extend_padding_mask(padding_mask=padding_mask, embeddings=embeddings)
		if self._causal:
			attn_mat.masked_fill_(self.causal_attn_mask[:, :, :max_length, :max_length] == 0, -1e4)
		# attn_probs: (batch_size, num_heads, max_length, max_length)
		attn_probs = self.softmax(attn_mat)
		attn_probs = self.dropout(attn_probs)

		# ctx_vectors: (batch_size, num_heads, max_length, per_head_dim)
		#    .permute: (batch_size, max_length, num_heads, per_head_dim)
		#    .view   : (batch_size, max_length, num_heads * per_head_dim)
		ctx_vectors = torch.matmul(attn_probs, sliced_projected_value).permute(0, 2, 1, 3).contiguous().view(batch_size, max_length, -1)
		attn_output = self.wo(ctx_vectors)
		attn_output = self.dropout(attn_output)

		return self.layer_norm(attn_output + embeddings), attn_probs

#test_mha_obj = MultiHeadAttention(test_config)
#test_padding_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
#test_mha_emb, test_mha_filters = test_mha_obj(test_emb, padding_mask=test_padding_mask)
#print_params(test_mha_obj)
#test_mha_emb, test_mha_emb.shape


class TogepiMultiHeadAttention(nn.Module):
	def __init__(self, config):
		super().__init__()

		assert(config.embedding_dim % config.num_attn_heads == 0)

		self._num_heads = config.num_attn_heads
		self._per_head_dim = config.embedding_dim // config.num_attn_heads
		max_length = config.max_position_embeddings - 1  # one position reserved for pad position
		self._training_max_length = max_length

		# out_features: (num_heads * per_head_dim)
		self.pre_proj = nn.Linear(in_features=config.embedding_dim, out_features=config.embedding_dim)
		self.pre_sparse_proj = nn.Linear(in_features=config.embedding_dim, out_features=config.embedding_dim)
		nn.init.xavier_normal_(self.pre_proj.weight.data)
		nn.init.xavier_normal_(self.pre_sparse_proj.weight.data)

		# randomly initialize point-spread functions, one per head
		# psf: [tok_weight, [tok_-1_weights, tok_-2_weight, ...], [..., tok_+2_weight, tok_+1_weight]]
		self.toeplitz_psfs = nn.Parameter(torch.randn(self._num_heads, 2 * max_length - 1, self._per_head_dim))
		self.attn_actn = F.gelu if config.attn_actn == 'gelu' else F.relu
		self.post_conv_proj = nn.Linear(in_features=config.embedding_dim, out_features=config.embedding_dim)
		nn.init.xavier_normal_(self.toeplitz_psfs.data)
		nn.init.xavier_normal_(self.post_conv_proj.weight.data)
		
		num_nonzero = int(max_length * max_length * config.sparse_dens)
		sparse_idxs = torch.randint(0, max_length, (num_nonzero, 2))
		sparse_vals = torch.randn(num_nonzero)
		self.sparse = nn.Parameter(torch.sparse_coo_tensor(sparse_idxs.t(), sparse_vals.abs(), size=(max_length, max_length)).to_dense())

		self._causal = config.causal_attn
		if config.causal_attn:
			# causal_psf_mask: ignore the tokens appearing ahead of the current token.
			self.register_buffer('causal_psf_mask', torch.tensor([1] + [1] * (max_length - 1) + [0] * (max_length - 1)).unsqueeze(0).unsqueeze(2))
			self.register_buffer('causal_sparse_mask', torch.tril(torch.ones(max_length, max_length)))
		
		self.layer_norm = nn.LayerNorm(normalized_shape=config.embedding_dim, eps=1e-12)
		self.dropout = nn.Dropout(p=config.attn_dropout_proba)

	def forward(self, embeddings, padding_mask=None, softmax_psf_weights=True):
		# embeddings: (batch_size, max_length, embedding_dim)
		# padding_mask: (batch_size, max_length)
		batch_size = embeddings.shape[0]
		max_length = embeddings.shape[1]
		embedding_dim = embeddings.shape[2]

		# expanded_padding_mask: (batch_size, max_length, 1)
		# 1: no pad, 0: pad
		expanded_padding_mask = None
		if padding_mask is not None:
			expanded_padding_mask = padding_mask.unsqueeze(2)

		# pre_proj_emb: (batch_size, max_length, num_heads * per_head_dim)
		pre_proj_emb = self.pre_proj(embeddings)
		if padding_mask is not None:
			pre_proj_emb.masked_fill_(expanded_padding_mask == 0, 0)
		# padded_embeddings: (batch_size, 2 * max_length - 1, embedding_dim)
		# F.pad: pad=(padding_left, padding_right, padding_top, padding_bottom)
		pre_proj_padded_embeddings = F.pad(pre_proj_emb, pad=(0, 0, 0, max_length - 1), mode='constant')
		# pre_proj_padded_embeddings: (batch_size, num_heads, 2 * max_length - 1, per_head_dim)
		pre_proj_padded_embeddings = pre_proj_padded_embeddings.view(batch_size, 2 * max_length - 1, self._num_heads, self._per_head_dim).permute(0, 2, 1, 3)

		psfs_weights = self.toeplitz_psfs.data
		if self._causal:
			if self._training_max_length == max_length:
				psfs_weights.masked_fill_(self.causal_psf_mask == 0, 0)
			else:
				# at inference time, the max_length changes per prompt
				causal_psf_mask = torch.tensor([1] + [1] * (max_length - 1) + [0] * (max_length - 1)).unsqueeze(0).unsqueeze(2)
				psfs_weights.masked_fill_(causal_psf_mask == 0, 0)
		if softmax_psf_weights:
			psfs_weights = F.softmax(psfs_weights, dim=1)
		psfs_fft = torch.fft.fftn(psfs_weights, dim=(1, 2))
		emb_fft = torch.fft.fftn(pre_proj_padded_embeddings, dim=(2, 3))
		# conv_output: (batch_size, num_heads, max_length, per_head_dim)
		conv_output = torch.real(torch.fft.ifftn(psfs_fft * emb_fft, dim=(2, 3))[:, :, :max_length, :])
		# conv_output: (batch_size, max_length, num_heads * per_head_dim)
		conv_output = self.attn_actn(conv_output).permute(0, 2, 1, 3).reshape(batch_size, max_length, -1)
		conv_emb = self.post_conv_proj(conv_output)
		
		
		sparse_data = self.sparse.data
		if self._causal:
			sparse_data.masked_fill_(self.causal_sparse_mask[:max_length, :max_length] == 0, 0)
		pre_sparse_emb = self.pre_sparse_proj(pre_proj_emb)
		if padding_mask is not None:
			pre_sparse_emb.masked_fill_(expanded_padding_mask == 0, 0)
		sparse_emb = torch.matmul(sparse_data, pre_sparse_emb)

		togepi_emb = self.dropout(conv_emb + sparse_emb)
		return self.layer_norm(togepi_emb + embeddings)
		
#test_togepi_mha_obj = TogepiMultiHeadAttention(test_config)
#test_padding_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
#test_togepi_mha_emb = test_togepi_mha_obj(test_emb, padding_mask=test_padding_mask)
#print_params(test_togepi_mha_obj)
#test_togepi_mha_emb, test_togepi_mha_emb.shape

def create_sparse_mat(sparse_dens=0.3, max_length=512):
	num_nonzero = int(max_length * max_length * sparse_dens)
	sparse_idxs = torch.randint(0, max_length, (num_nonzero, 2))
	sparse_vals = torch.randn(num_nonzero)
	return torch.sparse_coo_tensor(sparse_idxs.t(), sparse_vals.abs(), size=(max_length, max_length))

def create_emb(batch_size=32, max_length=512, embedding_dim=768):
	return torch.randn(batch_size, max_length, embedding_dim)

def sparse_matmul(sparse_mat, emb):
	# sparse_mat: (max_length, max_length) 
	batch_size, max_length, embedding_dim = emb.shape
	return torch.sparse.mm(sparse_mat, emb.permute(1, 0, 2).reshape(max_length, -1)).view(max_length, batch_size, -1).permute(1, 0, 2)

def sparse_to_dense_matmul(sparse_mat, emb):
	# sparse_mat: (max_length, max_length)
	return torch.matmul(sparse_mat.to_dense(), emb)

def dense_matmul(dense_mat, emb):
	return torch.matmul(dense_mat, emb)

# https://aclanthology.org/2021.emnlp-main.831.pdf
@dataclass(init=False)
class SpeedTestConfig:
	# embedding
	vocab_size: int
	padding_idx: int
	max_position_embeddings: int
	pad_position: int
	num_token_types: int
	pad_token_type: int
	embedding_dim: int
	embedding_dropout_proba: float
	causal_attn: bool
	num_attn_heads: int
	attn_actn: str
	sparse_dens: float
	attn_dropout_proba: float
	batch_size: int

	def __init__(self,max_position_embeddings,embedding_dim,num_attn_heads):
		self.vocab_size = 30522
		self.padding_idx = 0
		self.max_position_embeddings = max_position_embeddings+1
		self.pad_position = 0
		self.num_token_types = 3
		self.pad_token_type = 0
		self.embedding_dim = embedding_dim
		self.embedding_dropout_proba = 0.1
		
		# attention
		self.causal_attn = True  # for generative pre-training
		self.num_attn_heads = num_attn_heads
		self.attn_actn = 'gelu'
		self.sparse_dens = 0.3
		self.attn_dropout_proba = 0.1

		# training
		self.batch_size = 64
	
#test_speed_config = SpeedTestConfig()
#test_speed_config.vocab_size

def main():
	nub_heads = [2,8,16,64]
	#nub_heads = [2,4,8,16]
	test_max_position_embeddings = np.array([64,128,256,512,1024,2048,4096])
	test_embedding_dim = np.array([64,128,256,512,1024,2048,4096])
	n_col = test_max_position_embeddings.shape[0]
	n_row = test_embedding_dim.shape[0]
	nb_episodes = 7

	#mha_results_2 = np.zeros((n_row,n_col))
	#mha_results_8 = np.zeros((n_row,n_col))
	#mha_results_16 = np.zeros((n_row,n_col))
	#mha_results_64 = np.zeros((n_row,n_col))
	#togepi_result_2 = np.zeros((n_row,n_col))
	#togepi_result_8 = np.zeros((n_row,n_col))
	#togepi_result_16 = np.zeros((n_row,n_col))
	#togepi_result_64 = np.zeros((n_row,n_col))

	mha_results = np.zeros((n_row,n_col))
	togepi_result = np.zeros((n_row,n_col))

	for i in range(len(nub_heads)):
		for j in range(len(test_max_position_embeddings)):
			for k in range(len(test_embedding_dim)):
				test_speed_config = SpeedTestConfig(max_position_embeddings=test_max_position_embeddings[j]\
					,embedding_dim=test_embedding_dim[k],num_attn_heads=nub_heads[i])
				test_input_ids = torch.randint(low=0, high=test_speed_config.max_position_embeddings - 1, \
					size=(test_speed_config.batch_size, test_speed_config.max_position_embeddings - 1))

				test_emb_obj = Embedding(test_speed_config)
				test_emb = test_emb_obj(test_input_ids)

				test_mha_obj = MultiHeadAttention(test_speed_config)
				test_togepi_mha_obj = TogepiMultiHeadAttention(test_speed_config)

				mha_results[k,j] = 0
				togepi_result[k,j] = 0

				for _ in range(nb_episodes):
					start = time.time()
					test_mha_emb, test_mha_filters = test_mha_obj(test_emb)
					end = time.time()
					mha_results[k,j] += end - start
				mha_results[k,j] = mha_results[k,j]/nb_episodes

				for _ in range(nb_episodes):
					start = time.time()
					ttest_togepi_mha_emb = test_togepi_mha_obj(test_emb)
					end = time.time()
					togepi_result[k,j] += end - start
				togepi_result[k,j] = togepi_result[k,j]/nb_episodes

		file_name = "Results_Heads_"+str(nub_heads[i])+"_mha"+".txt"
		f = open("/mnt/beegfs/bulk/stripe/lm865/TimeResults/"+file_name,'wb')
		#f = open(file_name,'wb')
		for line in np.matrix(mha_results):
			np.savetxt(f,line,fmt='%.2f')
		f.close()
		file_name = "Results_Heads_"+str(nub_heads[i])+"_togepi"+".txt"
		f = open("/mnt/beegfs/bulk/stripe/lm865/TimeResults/"+file_name,'wb')
		#f = open(file_name,'wb')
		for line in np.matrix(togepi_result):
			np.savetxt(f,line,fmt='%.2f')
		f.close()
	file_name = "col_idx_test_max_position_embeddings.txt"
	f = open("/mnt/beegfs/bulk/stripe/lm865/TimeResults/"+file_name,'wb')
	#f = open(file_name,'wb')
	np.savetxt(f, test_max_position_embeddings, fmt='%.2f')
	f.close()

	file_name = "col_idx_test_embedding_dim.txt"
	f = open("/mnt/beegfs/bulk/stripe/lm865/TimeResults/"+file_name,'wb')
	#f = open(file_name,'wb')
	np.savetxt(f, test_embedding_dim, fmt='%.2f')
	f.close()


if __name__ == "__main__":
	main()



































