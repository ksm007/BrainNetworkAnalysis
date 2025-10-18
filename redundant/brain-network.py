import scipy.io

mat_data = scipy.io.loadmat('mat_subjects/M87102217_fiber.mat')
print(mat_data.keys())

for key in mat_data.keys():
    if not key.startswith("__"):
        print(f"{key}:{mat_data[key].shape} , type = {type(mat_data[key])}")
        
connectivity = mat_data['fibergraph']
print(connectivity)