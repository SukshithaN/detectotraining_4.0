from detecto import core

Train_dataset=core.Dataset(r'C:\Users\REKHA\Desktop\arun\images22')
# Train_dataset = core.Dataset(r'C:\Users\ANT-PC\Downloads\labelimg\CASE COMP BATTERY 17232-KVN-9700')

loader=core.DataLoader(Train_dataset, batch_size=1, shuffle=True)

# model = core.Model(['CASE COMP BATTERY 17232-KVN-9700'])
model = core.Model(["tank"])
model.fit(loader, epochs=2, lr_step_size=5, learning_rate=0.001, verbose=True)
model.save(r'D:\DETECTO CODE\det\defult.pth')





























