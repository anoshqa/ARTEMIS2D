def filetest(mipfilenames, mask_filenames):
    matched=[]
    unmatched=[]
    for i, ele in enumerate(mipfilenames):
        if ele[:30] == mask_filenames[i][:30]:
            matched.append(i)
        else:
            unmatched.append(i)
    if len(matched)==len(mipfilenames):
        return True
    else:
        return False