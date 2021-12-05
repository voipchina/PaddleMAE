import paddle

class MAE(paddle.nn.Layer):
    def __init__(self,image_size = 224, patch_size = 16, inch = 3, masking_ratio = 0.75):
        super(MAE, self).__init__()
        self.patch_size = patch_size
        self.embeddim = patch_size * patch_size * inch #768
        self.decoderdim = self.embeddim
        self.num_patches = (image_size // patch_size) ** 2
        self.embedproj = paddle.nn.Linear(self.embeddim, self.embeddim)
        self.encoder_layer = paddle.nn.TransformerEncoderLayer(d_model=self.embeddim, nhead=12, dim_feedforward=3072, dropout=0.1, activation='gelu', normalize_before = True)
        self.encoder = paddle.nn.TransformerEncoder(self.encoder_layer,num_layers = 12)
        self.patchproj = paddle.nn.Conv2D(inch, self.embeddim, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = paddle.create_parameter(shape=[1, self.num_patches, self.embeddim], dtype='float32',default_initializer = paddle.nn.initializer.TruncatedNormal(std=0.02))
        
        self.decoder_layer = paddle.nn.TransformerEncoderLayer(d_model=self.decoderdim, nhead=12, dim_feedforward=3072, dropout=0.1, activation='gelu', normalize_before = True)
        self.decoder = paddle.nn.TransformerEncoder(self.decoder_layer,num_layers = 2)  
        self.mask_token = paddle.create_parameter(shape=[self.embeddim], dtype='float32')
        self.masking_ratio = masking_ratio
        self.enc_dec = paddle.nn.Linear(self.embeddim, self.embeddim)

        self.outproj = paddle.nn.Linear(self.embeddim, self.embeddim)


    def forward(self, x):
        B, C, H, W = x.shape
        input = x

        x = self.patchproj(x)
        x = x.flatten(2)
        x = x.transpose(perm=(0,2,1)) #[8, 196, 768]

        num_masked = int(self.masking_ratio * self.num_patches)
        num_unmasked = self.num_patches - num_masked
        rand_indices = paddle.rand((B, self.num_patches)).argsort(axis = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        batch_range = paddle.arange(B)[:, None] #to [[0][1][2][3]]

        unmasked_embeddings = paddle.gather(
                x, unmasked_indices[0], axis=1)
        
        unmasked_positions = paddle.gather(
                self.pos_embedding, unmasked_indices[0], axis=1)

        masked_positions = paddle.gather(
                self.pos_embedding, masked_indices[0], axis=1)

        pos_embedding = paddle.expand(self.pos_embedding,shape=[B,self.num_patches, self.embeddim])
        unmasked_embeddings = x[batch_range,unmasked_indices]
        unmasked_positions = pos_embedding[batch_range,unmasked_indices]
        masked_positions = pos_embedding[batch_range,masked_indices]

        unmask_mixed = unmasked_embeddings + unmasked_positions

        encout = self.encoder(unmask_mixed)

        enc_dec = self.enc_dec(encout)


        mask_token = paddle.expand(self.mask_token,shape=[B,num_masked,self.embeddim])

        mask_mixed = mask_token + masked_positions

        totalmask = rand_indices
        totalmask = paddle.argsort(totalmask,axis = -1)
        #print (totalmask[0])
        totalembed = paddle.concat([mask_mixed, enc_dec], axis = 1)
        #print ("totalmask:", totalmask.shape)
        #print ("totalembed:",totalembed.shape)

        totalembed = totalembed[batch_range,totalmask]

        decinput = totalembed

        decout = self.decoder(decinput)
        
        decout = self.outproj(decout)

        displayout = decout.transpose(perm=(0,2,1))
        displayout = decout
        #print (decout.shape)

        target = input.reshape(
            [B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size]
        )  # Bx3x224x224 --> Bx3x16x14x16x14
        # Bx3x14x16x14x16 --> Bx(14*14)x(16*16*3)
        target = target.transpose([0, 2, 4, 3, 5, 1])
        target = target.reshape([B, self.num_patches, -1, C])

        target = target.reshape([B, self.num_patches, -1])

        #target = paddle.gather(target,unmasked_indices[0], axis = 1)
        #accout = paddle.gather(decout, unmasked_indices[0], axis = 1)

        target = target[batch_range, masked_indices]
        accout = decout[batch_range, masked_indices]

        # print (target.shape)
        # print (accout.shape)
        # print (displayout.shape)
        loss = paddle.nn.MSELoss()(accout.flatten(),target.flatten())

        #decout = paddle.concat([unmasked_embeddings,mask_token],axis = 1)
        displayout = decout.reshape([B, self.num_patches, -1, C]) # only for visual later 但是需要unshuffle....
        displayout = displayout.reshape([B,self.patch_size,self.patch_size, H // self.patch_size,W // self.patch_size, -1])
        displayout = displayout.transpose([0,5,1,3,2,4])
        displayout = displayout.reshape([B,C,H,W])

        #loss = paddle.nn.MSELoss()(displayout.flatten(),input.flatten())
        return encout, displayout, unmasked_indices, loss
