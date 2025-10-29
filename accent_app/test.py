
import os
p='pretrained_models/accent-id-commonaccent_ecapa'
for root,_,files in os.walk(p):
    for f in files:
        fp=os.path.join(root,f)
        print(fp, "islink?", os.path.islink(fp))