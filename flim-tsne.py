import cudf
from cuml.manifold import TSNE
import matplotlib.pyplot as plt
from cuml.preprocessing import StandardScaler
import cupy as cp

filename = 'TSNEtest.txt'
df = cudf.read_csv(filename)

treatment = df['Treatment']
color_dict = {'dox0': 'red', 'dox20': 'blue', 'dox40' : 'green', 'dox60': 'purple'}
colors = treatment.map(color_dict)

colors_np = colors.to_numpy()

df = df.drop(df.columns[0], axis=1)

# Drop irrelevant columns
df = df.drop(['Directory','File','Cell','FOV','Treatment','FAD a1[%]/a2[%]','FAD chi','FAD offset','FAD photons/NAD(P)H photons','FAD scatter','FAD shift','FAD tm','FLIRR','NAD(P)H a2[%]/a1[%]','NAD(P)H chi','NAD(P)H offset','NAD(P)H scatter','NAD(P)H shift','NAD(P)H tm','NAD(P)H tm/FAD tm','NADH %','NADPH %','NADPH a2/FAD a1'], axis = 1)

# Features generally follow a normal distrubution. Of the 14 around 2-3 clearly follow different distributions
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

tsne = TSNE(n_components = 2, perplexity = 30, learning_rate = 200, n_iter = 1000) 
output = tsne.fit_transform(scaled_data)

output_np = output.to_numpy()

plt.scatter(output_np[:, 0], output_np[:,1], c = colors_np, s = 0.5)
plt.title("TSNE Plot of FLIM Data Collected from HeLa Cells Treated with Doxorubicin at Four Time Points")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()

