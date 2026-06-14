def matchtest(images, masks):
    matched=[]
    unmatched=[]
    for i in range(len(images)):
        if images[i].shape == masks[i].shape:
            matched.append(i)
        else:
            unmatched.append(i)
    if len(matched)==len(images):
        return True
    else:
        return False