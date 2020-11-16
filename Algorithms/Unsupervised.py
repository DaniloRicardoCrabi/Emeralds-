from sklearn.cluster import KMeans
import pandas as pd

dataset = pd.read_csv("featuresEmeralds.csv")

data = dataset[dataset.columns[1:dataset.shape[1]-1]]

norm = (data - data.mean())/data.std()
data = norm

cl1 = data[0:24]
cl2 = data[24:48]
cl3 = data[48:72]
cl4 = data[72:96]
cl5 = data[96:120]
cl6 = data[120:144]
cl7 = data[144:168]
cl8 = data[168:192]

cl1 = cl1.sample(frac=1, random_state=42)
cl2 = cl2.sample(frac=1, random_state=42)
cl3 = cl3.sample(frac=1, random_state=42)
cl4 = cl4.sample(frac=1, random_state=42)
cl5 = cl5.sample(frac=1, random_state=42)
cl6 = cl6.sample(frac=1, random_state=42)
cl7 = cl7.sample(frac=1, random_state=42)
cl8 = cl8.sample(frac=1, random_state=42)


te11 = cl1[0:8]
tr11 = cl1[8:24]
te12 = cl1[8:16]
te13 = cl1[16:24]
tr13 = cl1[0:16]
tr12 = pd.concat([te11, te13])

te11 = cl1[0:8]
tr11 = cl1[8:24]
te12 = cl1[8:16]
te13 = cl1[16:24]
tr13 = cl1[0:16]
tr12 = pd.concat([te11, te13])

te21 = cl2[0:8]
tr21 = cl2[8:24]
te22 = cl2[8:16]
te23 = cl2[16:24]
tr23 = cl2[0:16]
tr22 = pd.concat([te21, te23])

te31 = cl3[0:8]
tr31 = cl3[8:24]
te32 = cl3[8:16]
te33 = cl3[16:24]
tr33 = cl3[0:16]
tr32 = pd.concat([te31, te33])

te41 = cl4[0:8]
tr41 = cl4[8:24]
te42 = cl4[8:16]
te43 = cl4[16:24]
tr43 = cl4[0:16]
tr42 = pd.concat([te41, te43])

te51 = cl5[0:8]
tr51 = cl5[8:24]
te52 = cl5[8:16]
te53 = cl5[16:24]
tr53 = cl5[0:16]
tr52 = pd.concat([te51, te53])

te61 = cl6[0:8]
tr61 = cl6[8:24]
te62 = cl6[8:16]
te63 = cl6[16:24]
tr63 = cl6[0:16]
tr62 = pd.concat([te61, te63])

te71 = cl7[0:8]
tr71 = cl7[8:24]
te72 = cl7[8:16]
te73 = cl7[16:24]
tr73 = cl7[0:16]
tr72 = pd.concat([te71, te73])

te81 = cl8[0:8]
tr81 = cl8[8:24]
te82 = cl8[8:16]
te83 = cl8[16:24]
tr83 = cl8[0:16]
tr82 = pd.concat([te81, te83])

teste1 = pd.concat([te11, te21, te31, te41, te51, te61, te71, te81])
teste2 = pd.concat([te12, te22, te32, te42, te52, te62, te72, te82])
teste3 = pd.concat([te13, te23, te33, te43, te53, te63, te73, te83])

treino1 = pd.concat([tr11, tr21, tr31, tr41, tr51, tr61, tr71, tr81])
treino2 = pd.concat([tr12, tr22, tr32, tr42, tr52, tr62, tr72, tr82])
treino3 = pd.concat([tr13, tr23, tr33, tr43, tr53, tr63, tr73, tr83])


k = 8
model1 = KMeans(n_clusters=k, max_iter=1000, algorithm='auto', random_state=42)
model1.fit(treino1)

model2 = KMeans(n_clusters=k, max_iter=1000, algorithm='auto', random_state=42)
model2.fit(treino2)

model3 = KMeans(n_clusters=k, max_iter=1000, algorithm='auto', random_state=42)
model3.fit(treino3)

all_predictions1 = model1.predict(teste1)
all_predictions2 = model2.predict(teste2)
all_predictions3 = model3.predict(teste3)

clusters1 = []

for j in range(0, k):
    for i in range(0, teste1.shape[0]):
        clusters1.append([])
        if all_predictions1[i] == j:
            clusters1[j].append(i)

clusters2 = []

for j in range(0, k):
    for i in range(0, teste2.shape[0]):
        clusters2.append([])
        if all_predictions2[i] == j:
            clusters2[j].append(i)

clusters3 = []

for j in range(0, k):
    for i in range(0, teste3.shape[0]):
        clusters3.append([])
        if all_predictions3[i] == j:
            clusters3[j].append(i)
