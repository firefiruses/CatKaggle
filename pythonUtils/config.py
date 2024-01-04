
size = 128

grayscale = True

normalization = False

train_split_size = 0.3

mean = [124.5304, 111.7974,  98.9443]
std = [69.3189, 67.4342, 68.1056]

intresting_pictures = ['2a5e97796bb037c190f9eafe4fb60ac2.jpg', '0acec983f7d559f971a8e38b90847cd5.jpg', '0e977d2dcff9a4b5e8913fd8ae90a3cb.jpg', '2b5a1ddb0d4444cc06a293f98ea61a48.jpg', '2bcb4c75bc83ed168dde3150f2d1b982.jpg', '2e3946c777c6f0b7e99f5c86afd78b29.jpg', '3af893700f3cd3a4a5acb80043bb4dd2.jpg', '4aeea389f06b3c13c6a4821817d7c7ba.jpg', '4b234dcae2b8ab10206dc57665e8decd.jpg', '4d4d795d784f3a8d3d07479e0ac79073.jpg']

def get_s():
    s = ""
    if grayscale:
        s = s + "_gs"
    if normalization:
        s = s + "_norm"
    s = s + "_" + str(size)
    return s