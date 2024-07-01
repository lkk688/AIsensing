import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#mask (0,1 value) two squares
mask_tf=np.load('mask_tf.npy') #(1, 2, 896)
mask=np.load('mask.npy') #(1, 2, 896)
plt.figure()
plt.plot(mask_tf[0,0,:])
plt.plot(mask[0,0,:], 'r-')
plt.title('mask(1, 2, 896)')
print(np.allclose(mask_tf, mask))

print(mask[0,0,:])
num_pilot_symbols = 128
pilot_ind = tf.argsort(mask, axis=-1, direction="DESCENDING") #(1, 2, 896) descending order (from largest to smallest). 
print("pilot_ind", pilot_ind[0,0,:]) #return the index, two sqaures (128-191, 704-767) rank to the top, other 0s after (0-127, 192-703, 768-895) 
pilot_ind_new = pilot_ind[...,:num_pilot_symbols] #only select index (128-191, 704-767) with 1s in mask
print("pilot_ind_new", pilot_ind_new[0,0,:])

print("mask:",mask[0,0,:])
# pilot_ind_new2 = np.argsort(mask) #np.argsort is small to bigger, np.argsort(mask, axis=-1)#[..., ::-1] #(1, 2, 896) reverses the order of the indices along the last axis
# print("pilot_ind_new2", pilot_ind_new2[0,0,:])
mask_array = np.array(mask)
pilot_ind_new2 = np.where(mask_array == 1)[0]
pilot_ind_new2 = pilot_ind_new2[...,:num_pilot_symbols] #(1, 2, 128)
print("pilot_ind_new2b", pilot_ind_new2[0,0,:])
print(np.allclose(pilot_ind_new, pilot_ind_new2))
plt.figure()
plt.plot(pilot_ind_new[0,0,:])
plt.plot(pilot_ind_new2[0,0,:], 'r-')
plt.title('pilot_ind_new(1, 2, 128)')

pilot_ind_tf=np.load('pilot_ind_tf.npy') #(1, 2, 128)
y_eff_flat_tf=np.load('y_eff_flat_tf.npy') #(2, 1, 16, 896)
y_pilots_tf=np.load('y_pilots_tf.npy') #(2, 1, 16, 1, 2, 128)

pilot_ind=np.load('pilot_ind.npy') #(1, 2, 128)
y_eff_flat=np.load('y_eff_flat.npy') #(2, 1, 16, 896)
y_pilots=np.load('y_pilots.npy') #(2, 1, 16, 1, 2, 128)

plt.figure()
plt.plot(pilot_ind_tf[0,0,:])
plt.plot(pilot_ind[0,0,:], 'r-')
plt.title('pilot_ind(1, 2, 128)')

plt.figure()
plt.plot(np.real(y_eff_flat_tf[0,0,0,:]))
plt.plot(np.imag(y_eff_flat_tf[0,0,0,:]))
plt.plot(np.real(y_eff_flat[0,0,0,:]), 'r-')
plt.plot(np.imag(y_eff_flat[0,0,0,:]), 'g-')
plt.title('y_eff_flat(2, 1, 16, 896)')

# plt.figure()
# plt.plot(np.real(y_pilots_tf[0,0,0,0,0,:]))
# plt.plot(np.real(y_pilots[0,0,0,0,0,:]), 'r-')


plt.figure()
plt.plot(np.real(y_pilots_tf[0,0,0,0,0,:]))
plt.plot(np.imag(y_pilots_tf[0,0,0,0,0,:]))
plt.plot(np.real(y_pilots[0,0,0,0,0,:]), 'r-')
plt.plot(np.imag(y_pilots[0,0,0,0,0,:]), 'g-')
plt.title('y_pilots(2, 1, 16, 1, 2, 128)')
# Given shapes
shape_y_eff_flat = (2, 1, 16, 896)
shape_pilot_ind = (1, 2, 128)

# Create random data for y_eff_flat and pilot_ind (you can replace with actual data)
y_eff_flat = np.random.uniform(-1, 1, shape_y_eff_flat) + 1.j * np.random.uniform(-1, 1, shape_y_eff_flat) #np.random.rand(*shape_y_eff_flat)
pilot_ind = np.random.randint(0, 896, size=shape_pilot_ind)

#y_eff_flat:(2, 1, 16, 896), _pilot_ind:(1, 2, 128) => y_pilots(2, 1, 16, 1, 2, 128)
y_eff_flat_tf=tf.convert_to_tensor(y_eff_flat, dtype=tf.complex64) #(2, 1, 2, 1148)
#pilot_ind_tf=tf.convert_to_tensor(pilot_ind, dtype=tf.complex64) 
y_pilots_tf = tf.gather(y_eff_flat_tf, pilot_ind, axis=-1)
print(y_pilots_tf.shape) #y_pilots(2, 1, 16, 1, 2, 128)

y_pilots1 = np.take(y_eff_flat, pilot_ind, axis=-1)
print(y_pilots1.shape)

y_pilots2 = y_eff_flat[..., pilot_ind]
print(y_pilots2.shape)

plt.figure()
plt.plot(np.real(y_pilots_tf[0,0,0,0,0,:]))
plt.plot(np.real(y_pilots1[0,0,0,0,0,:]), 'r-')
plt.plot(np.real(y_pilots2[0,0,0,0,0,:]), 'g-')
plt.title('Compare')
plt.show()

plt.figure()
plt.plot(np.imag(y_pilots_tf[0,0,0,0,0,:]))
plt.plot(np.imag(y_pilots1[0,0,0,0,0,:]), 'r-')
plt.plot(np.imag(y_pilots2[0,0,0,0,0,:]), 'g-')
plt.title('Compare')
plt.show()