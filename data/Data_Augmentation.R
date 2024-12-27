library(GenomicRanges)  # 加载GenomicRanges库，用于处理基因组范围数据
library(BSgenome.Hsapiens.UCSC.hg19.masked)  # 加载人类基因组数据，用于获取序列信息

#Delete the files before running the script
#GM12878, HeLa-S3, HUVEC, IMR90, K562, NHEK
celline <- "../src/data/GM12878"  # 设置细胞系为IMR90
folder <- "aug"  # 文件夹名称为aug
shift_distance <- 50  # 偏移距离设为50
random_seed <- 1  # 随机种子设为1
enhancer_target_length <- 3000  # 增强子目标长度设为3000
promoter_target_length <- 2000  # 启动子目标长度设为2000
positive_scalar <- 20  # 正样本倍增系数设为20
test_percent <- 0.1  # 测试集百分比设为0.1
pair_file_name <- paste("./data","/",celline,".csv",sep="")  # 组合出文件名

############################################################################################################
#balanced training 平衡训练
enhancer_file_name <- paste(folder,"/",celline,"_enhancer.fasta",sep="")
promoter_file_name <- paste(folder,"/",celline,"_promoter.fasta",sep="")
label_file_name <- paste(folder,"/",celline,"_label.txt", sep="")
enhancer_coor_file_name <- paste(folder,"/",celline,"_enhancer_coor.txt", sep="")
promoter_coor_file_name <- paste(folder,"/",celline,"_promoter_coor.txt", sep="")
enhancer_coor_file_name_stat <- paste(folder,"/",celline,"_enhancer_coor_stat.txt", sep="")
promoter_coor_file_name_stat <- paste(folder,"/",celline,"_promoter_coor_stat.txt", sep="")
enhancer_bed_file_name <- paste(folder,"/", celline, "_enhancer.bed", sep="")
promoter_bed_file_name <- paste(folder,"/", celline, "_promoter.bed", sep="")

#test
enhancer_file_name_test <- paste(folder,"/", celline, "_enhancer_test.fasta", sep="")
promoter_file_name_test <- paste(folder,"/", celline, "_promoter_test.fasta", sep="")
label_file_name_test <- paste(folder,"/", celline, "_label_test.txt", sep="")
enhancer_coor_file_name_test <- paste(folder,"/", celline, "_enhancer_coor_test.txt", sep="")
promoter_coor_file_name_test <- paste(folder,"/", celline, "_promoter_coor_test.txt", sep="")
enhancer_bed_file_name_test <- paste(folder,"/",celline, "_enhancer_test.bed", sep="")
promoter_bed_file_name_test <- paste(folder,"/", celline, "_promoter_test.bed",sep="")

#imbalanced training
im_folder <- paste(folder,"/imbalanced/", sep="")
dir.create(im_folder)

im_enhancer_file_name <- paste(im_folder,celline,"_enhancer.fasta",sep="")
im_promoter_file_name <- paste(im_folder,celline,"_promoter.fasta",sep="")
im_label_file_name <- paste(im_folder,celline,"_label.txt", sep="")
im_enhancer_coor_file_name <- paste(im_folder,celline,"_enhancer_coor.txt", sep="")
im_promoter_coor_file_name <- paste(im_folder,celline,"_promoter_coor.txt", sep="")
im_enhancer_coor_file_name_stat <- paste(im_folder,celline,"_enhancer_coor_stat.txt", sep="")
im_promoter_coor_file_name_stat <- paste(im_folder,celline,"_promoter_coor_stat.txt", sep="")
im_enhancer_bed_file_name <- paste(im_folder, celline, "_enhancer.bed", sep="")
im_promoter_bed_file_name <- paste(im_folder, celline, "_promoter.bed", sep="")

############################################################################################################
#adjustment for the negative part
adjustment <- function(chr_end, start, end, target_len){
  str_len <- end-start+1
  if (str_len < target_len){
    start_update <- start-ceiling((target_len-str_len)/2)
    if (start_update < 1){
      print("error")
    }
    end_update <- start_update+target_len-1
    
    if (end_update > chr_end){
      end_update <- chr_end
      start_update <- end_update+1-target_len
    }#if
    
    start_coor <- start-start_update+1
    end_coor <- start_coor+str_len-1
    
  }else if (str_len > target_len){
    start_update <- start+ ceiling((str_len - target_len)/2)
    end_update <- start_update+target_len-1
    start_coor <- 1
    end_coor <- target_len
  }else{
    start_update <- start
    end_update <- end
    start_coor <- 1
    end_coor <- str_len
  }
  return(list(start = start_update, end = end_update, start_coor = start_coor, end_coor = end_coor))  
}#adjustment

#adjustment for the positive part
# 定义丰富调整函数，用于增强正样本数据
enrich_adjustment <- function(chr_end, start, end, target_len, scalar, shift_distance){
  start_update <- c()  # 初始化存储调整后起始位置的向量
  end_update <- c()  # 初始化存储调整后终止位置的向量
  start_coor_update <- c()  # 初始化存储调整后起始坐标的向量
  end_coor_update <- c()  # 初始化存储调整后终止坐标的向量
  str_len <- end - start + 1  # 计算原始序列长度

  if (str_len < target_len) {  # 如果原始序列长度小于目标长度
    if (ceiling((end + start + target_len) / 2 + (scalar - scalar / 2) * shift_distance) - 1 > chr_end) {  # 如果计算后的终止位置超过染色体末端
      constant <- ceiling(((end + start + target_len) / 2 - chr_end - 1) / shift_distance) + scalar  # 计算常数值
    } else {  # 否则
      constant <- scalar / 2  # 常数值为标量的一半
    }

    for (i in 1:scalar) {  # 循环scalar次
      start_new <- ceiling((end + start - target_len) / 2 + (i - constant) * shift_distance)  # 计算新的起始位置
      end_new <- start_new + target_len - 1  # 计算新的终止位置

      start_update <- c(start_update, start_new)  # 将新的起始位置添加到向量中
      end_update <- c(end_update, end_new)  # 将新的终止位置添加到向量中
      start_coor_update <- c(start_coor_update, start - start_new + 1)  # 计算并存储新的起始坐标
      end_coor_update <- c(end_coor_update, start - start_new + str_len)  # 计算并存储新的终止坐标
    }# for
  } else {  # 如果原始序列长度大于等于目标长度
    for (i in 1:scalar) {  # 循环scalar次
      start_new <- ceiling((end + start - target_len) / 2 + (i - scalar / 2) * shift_distance)  # 计算新的起始位置
      end_new <- start_new + target_len - 1  # 计算新的终止位置
      start_update <- c(start_update, start_new)  # 将新的起始位置添加到向量中
      end_update <- c(end_update, end_new)  # 将新的终止位置添加到向量中
      start_coor_update <- c(start_coor_update, 1)  # 新的起始坐标为1
      end_coor_update <- c(end_coor_update, target_len)  # 新的终止坐标为目标长度
    }
  }# else

  return(list(start = start_update, end = end_update, start_coor = start_coor_update, end_coor = end_coor_update))  # 返回调整后的起始位置、终止位置、起始坐标和终止坐标
}# enrich_adjustment


#deal with N in the sequence
processN <- function(seq){
  #return(gsub("N", "A", seq))
  return(seq)
}#processN

############################################################################################################
pair <- read.csv(pair_file_name, header=TRUE)
interaction <- pair$label
enhancer_list <- as.character(pair$enhancer_name)
promoter_list <- as.character(pair$promoter_name)

interaction_positive <- which(interaction==1)
interaction_negative <- which(interaction==0)


#first create test file
############################################################################################################
############################################################################################################
set.seed(random_seed)

positive_test <- sort(sample(interaction_positive,length(interaction_positive)*test_percent))
negative_test <- sort(sample(interaction_negative,length(interaction_negative)*test_percent))
test_index <- c(positive_test, negative_test)
train_index <- setdiff(1:length(interaction),test_index)

interaction_test <- interaction[test_index]
enhancer_list_test <- enhancer_list[test_index]
promoter_list_test <- promoter_list[test_index]

#enhancer_test
chrs <- c()
starts <- c()
ends <- c()
starts_coor <- c()
ends_coor <- c()
locs <- c()

for (i in 1:length(interaction_test)){
  loc <- strsplit(enhancer_list_test[i],split="|", fixed=T)[[1]][2]
  chr <- strsplit(loc,  split = ":")[[1]][1]
  segment <- strsplit(loc,  split = ":")[[1]][2]
  start <- strsplit(segment, split = "-")[[1]][1]
  end <- strsplit(segment, split = "-")[[1]][2]
  start <- as.numeric(start)
  end <- as.numeric(end)
  
  chr_end <- seqlengths(Hsapiens)[chr]
  adj <- adjustment(chr_end, start, end, enhancer_target_length)
  
  start <- adj$start
  end <- adj$end
  start_coor <- adj$start_coor
  end_coor <- adj$end_coor
  
  chrs <- c(chrs, chr)
  starts <- c(starts, start)
  ends <- c(ends, end)
  starts_coor <- c(starts_coor, start_coor)
  ends_coor <- c(ends_coor, end_coor)
  locs <- c(locs, loc)
}#for i


#write bed files
write.table(cbind(interaction_test, chrs,starts,ends), file = enhancer_bed_file_name_test, sep="\t", row.names=F, col.names = F, quote = F)


rngs <- GRanges(chrs, IRanges(starts, ends))
seqs <- getSeq(Hsapiens, rngs)


#write enhancer_test string
str <- ""
for (i in 1:length(seqs)){
  #print(i)
  str_pre <- processN(toString(seqs[i]))
  str <- paste(">", locs[i], "\n", str_pre, sep="")    
  write(str,file = enhancer_file_name_test, append = T)
}


#write coordinates
write.table(cbind(starts_coor, ends_coor), file = enhancer_coor_file_name_test, sep="\t", row.names = F, col.names = F)





############################################################################################################
#promoter_test
chrs <- c()
starts <- c()
ends <- c()
starts_coor <- c()
ends_coor <- c()
locs <- c()

for (i in 1:length(interaction_test)){
  loc <- strsplit(promoter_list_test[i],split="|", fixed=T)[[1]][2]
  chr <- strsplit(loc,  split = ":")[[1]][1]
  segment <- strsplit(loc,  split = ":")[[1]][2]
  start <- strsplit(segment, split = "-")[[1]][1]
  end <- strsplit(segment, split = "-")[[1]][2]
  start <- as.numeric(start)
  end <- as.numeric(end)
  
  chr_end <- seqlengths(Hsapiens)[chr]
  adj <- adjustment(chr_end, start, end, promoter_target_length)
  start <- adj$start
  end <- adj$end
  start_coor <- adj$start_coor
  end_coor <- adj$end_coor
  
  chrs <- c(chrs, chr)
  starts <- c(starts, start)
  ends <- c(ends, end)
  starts_coor <- c(starts_coor, start_coor)
  ends_coor <- c(ends_coor, end_coor)
  locs <- c(locs, loc)
}#for i


#write bed files
write.table(cbind(interaction_test, chrs, starts, ends), file = promoter_bed_file_name_test, sep="\t", row.names=F, col.names = F, quote = F)


rngs <- GRanges(chrs, IRanges(starts, ends))
seqs <- getSeq(Hsapiens, rngs)

#write promoter_test string
str <- ""
for (i in 1:length(seqs)){
  #print(i)
  str_pre <- processN(toString(seqs[i]))
  str <- paste(">", locs[i], "\n", str_pre, sep="")    
  write(str,file = promoter_file_name_test, append = T)
}

#write coordinates
write.table(cbind(starts_coor, ends_coor), file = promoter_coor_file_name_test, sep="\t", row.names = F, col.names = F)


#labels
write(interaction_test, file = label_file_name_test, ncolumns = 1)










#create new interaction, enhancer_list, promoter_list
interaction <- interaction[train_index]
#augment interaction by positive_scalar
interaction_train <- c(rep(1, sum(interaction)*positive_scalar), rep(0, sum(interaction==0)))
  
enhancer_list <- enhancer_list[train_index]
promoter_list <- promoter_list[train_index]

interaction_positive <- which(interaction==1)
interaction_negative <- which(interaction==0)



#create training file
############################################################################################################
############################################################################################################
#first get enhancer sequences, and write it to fasta file
chrs <- c()
starts <- c()
ends <- c()
starts_coor <- c()
ends_coor <- c()
locs <- c()

im_chrs <- c()
im_starts <- c()
im_ends <- c()
im_starts_coor <- c()
im_ends_coor <- c()
im_locs <- c()

enhancer_length <- c()



for (i in 1:length(interaction)){
  #print(i)
  loc <- strsplit(enhancer_list[i],split="|", fixed=T)[[1]][2]
  chr <- strsplit(loc,  split = ":")[[1]][1]
  segment <- strsplit(loc,  split = ":")[[1]][2]
  start <- strsplit(segment, split = "-")[[1]][1]
  end <- strsplit(segment, split = "-")[[1]][2]
  start <- as.numeric(start)
  end <- as.numeric(end)
  
  #adjust the start and end based on the expectation
  enhancer_length <- c(enhancer_length, end-start+1)
  chr_end <- seqlengths(Hsapiens)[chr]
  if (interaction[i]==0){
    adj <- adjustment(chr_end, start, end, enhancer_target_length)
    start <- adj$start
    end <- adj$end
    start_coor <- adj$start_coor 
    end_coor <- adj$end_coor
    
    im_start <- start
    im_end <- end
    im_start_coor <- start_coor
    im_end_coor <- end_coor
    im_chr <- chr
    im_loc <- loc
    
    
  }else{
    start0 <- start
    end0 <- end
    
    adj <- enrich_adjustment(chr_end, start, end, enhancer_target_length, positive_scalar, shift_distance)
    start <- adj$start
    end <- adj$end
    start_coor <- adj$start_coor
    end_coor <-adj$end_coor
    #replicate locs, chrs
    chr <- rep(chr, positive_scalar)
    loc <- rep(loc, positive_scalar)
    
    
    adj <- adjustment(chr_end, start0, end0, enhancer_target_length)
    im_start <- adj$start
    im_end <- adj$end
    im_start_coor <- adj$start_coor
    im_end_coor <- adj$end_coor
    im_chr <- chr[1]
    im_loc <- loc[1]
  }#else
  
  
  chrs <- c(chrs, chr)
  starts <- c(starts, start)
  ends <- c(ends, end)
  starts_coor <- c(starts_coor, start_coor)
  ends_coor <- c(ends_coor, end_coor)
  locs <- c(locs, loc)
  
  im_chrs <- c(im_chrs, im_chr)
  im_starts <- c(im_starts, im_start)
  im_ends <- c(im_ends, im_end)
  im_starts_coor <- c(im_starts_coor, im_start_coor)
  im_ends_coor <- c(im_ends_coor, im_end_coor)
  im_locs <- c(im_locs, im_loc)
}#for interaction


#change the order: chrs, starts, ends
chrs_tem <- c()
starts_tem <- c()
ends_tem <- c()
starts_coor_tem <- c()
ends_coor_tem <- c()
locs_tem <- c()
len <- length(interaction_positive)

for (i in 1:positive_scalar){
  for (j in 1:len){
    chrs_tem <- c(chrs_tem, chrs[i+(j-1)*positive_scalar])
    starts_tem <- c(starts_tem, starts[i+(j-1)*positive_scalar])
    ends_tem <- c(ends_tem, ends[i+(j-1)*positive_scalar])
    starts_coor_tem <- c(starts_coor_tem, starts_coor[i+(j-1)*positive_scalar])
    ends_coor_tem <- c(ends_coor_tem, ends_coor[i+(j-1)*positive_scalar])
    locs_tem <- c(locs_tem, locs[i+(j-1)*positive_scalar])
  }#for j
}#for i


chrs[1:(positive_scalar*len)] <- chrs_tem
starts[1:(positive_scalar*len)] <- starts_tem
ends[1:(positive_scalar*len)] <- ends_tem
starts_coor[1:(positive_scalar*len)] <- starts_coor_tem
ends_coor[1:(positive_scalar*len)] <- ends_coor_tem
locs[1:(positive_scalar*len)] <- locs_tem



######################################################
#training
#write bed files
write.table(cbind(interaction_train, chrs,starts,ends), file = enhancer_bed_file_name, sep="\t", row.names=F, col.names = F, quote = F)


rngs <- GRanges(chrs, IRanges(starts, ends))
seqs <- getSeq(Hsapiens, rngs)


str <- ""
for (i in 1:length(seqs)){
  #print(i)
  str_pre <- processN(toString(seqs[i]))
  str <- paste(">", locs[i], "\n", str_pre, sep="")    
  write(str,file = enhancer_file_name, append = T)
}

#z-score
write.table(rbind(c(mean(starts_coor), sd(starts_coor)),c(mean(ends_coor), sd(ends_coor))), file = enhancer_coor_file_name_stat, sep="\t", row.names = F, col.names = F)

starts_coor <- scale(starts_coor)
ends_coor <- scale(ends_coor)

#write coordinates
write.table(cbind(starts_coor, ends_coor), file = enhancer_coor_file_name, sep="\t", row.names = F, col.names = F)


######################################################
#imbalanced training
write.table(cbind(interaction, im_chrs, im_starts, im_ends), file = im_enhancer_bed_file_name, sep="\t", row.names=F, col.names = F, quote = F)


rngs <- GRanges(im_chrs, IRanges(im_starts, im_ends))
seqs <- getSeq(Hsapiens, rngs)

str <- ""
for (i in 1:length(seqs)){
  #print(i)
  str_pre <- processN(toString(seqs[i]))
  str <- paste(">", im_locs[i], "\n", str_pre, sep="")    
  write(str,file = im_enhancer_file_name, append = T)
}

#z-score
write.table(rbind(c(mean(im_starts_coor), sd(im_starts_coor)),c(mean(im_ends_coor), sd(im_ends_coor))), file = im_enhancer_coor_file_name_stat, sep="\t", row.names = F, col.names = F)

im_starts_coor <- scale(im_starts_coor)
im_ends_coor <- scale(im_ends_coor)

#write coordinates
write.table(cbind(im_starts_coor, im_ends_coor), file = im_enhancer_coor_file_name, sep="\t", row.names = F, col.names = F)









############################################################################################################
#firt get promoter sequences, and write it to fasta file
#one generate twenty for positive

chrs <- c()
starts <- c()
ends <- c()
starts_coor <- c()
ends_coor <- c()
locs <- c()


im_chrs <- c()
im_starts <- c()
im_ends <- c()
im_starts_coor <- c()
im_ends_coor <- c()
im_locs <- c()


promoter_length <- c()

for (i in 1:length(interaction)){
  #print(i)
  loc <- strsplit(promoter_list[i],split="|", fixed=T)[[1]][2]
  chr <- strsplit(loc,  split = ":")[[1]][1]
  segment <- strsplit(loc,  split = ":")[[1]][2]
  start <- strsplit(segment, split = "-")[[1]][1]
  end <- strsplit(segment, split = "-")[[1]][2]
  start <- as.numeric(start)
  end <- as.numeric(end)
  
  #adjust the start and end based on the expectation
  promoter_length <- c(promoter_length, end-start+1)
  chr_end <- seqlengths(Hsapiens)[chr]
  if (interaction[i]==0){
    adj <- adjustment(chr_end, start, end, promoter_target_length)
    start <- adj$start
    end <- adj$end
    start_coor <- adj$start_coor
    end_coor <- adj$end_coor
    
    im_start <- start
    im_end <- end
    im_start_coor <- start_coor
    im_end_coor <- end_coor
    im_chr <- chr
    im_loc <- loc
    
  }else{
    start0 <- start
    end0 <- end
    
    adj <- enrich_adjustment(chr_end, start, end, promoter_target_length, positive_scalar, shift_distance)
    start <- adj$start
    end <- adj$end
    start_coor <- adj$start_coor
    end_coor <- adj$end_coor
    #replicate locs, chrs
    chr <- rep(chr, positive_scalar)
    loc <- rep(loc, positive_scalar)
    
    adj <- adjustment(chr_end, start0, end0, promoter_target_length)
    im_start <- adj$start
    im_end <- adj$end
    im_start_coor <- adj$start_coor
    im_end_coor <- adj$end_coor
    im_chr <- chr[1]
    im_loc <- loc[1]
  }#else
  
  
  chrs <- c(chrs, chr)
  starts <- c(starts, start)
  ends <- c(ends, end)
  starts_coor <- c(starts_coor, start_coor)
  ends_coor <- c(ends_coor, end_coor)
  locs <- c(locs, loc)
  
  im_chrs <- c(im_chrs, im_chr)
  im_starts <- c(im_starts, im_start)
  im_ends <- c(im_ends, im_end)
  im_starts_coor <- c(im_starts_coor, im_start_coor)
  im_ends_coor <- c(im_ends_coor, im_end_coor)
  im_locs <- c(im_locs, im_loc)
}#for interaction




#change the order: chrs, starts, ends
chrs_tem <- c()
starts_tem <- c()
ends_tem <- c()
starts_coor_tem <- c()
ends_coor_tem <- c()
locs_tem <- c()
len <- length(interaction_positive)


for (i in 1:positive_scalar){
  for (j in 1:len){
    chrs_tem <- c(chrs_tem, chrs[i+(j-1)*positive_scalar])
    starts_tem <- c(starts_tem, starts[i+(j-1)*positive_scalar])
    ends_tem <- c(ends_tem, ends[i+(j-1)*positive_scalar])
    starts_coor_tem <- c(starts_coor_tem, starts_coor[i+(j-1)*positive_scalar])
    ends_coor_tem <- c(ends_coor_tem, ends_coor[i+(j-1)*positive_scalar])
    locs_tem <- c(locs_tem, locs[i+(j-1)*positive_scalar])
  }#for j
}#for i


chrs[1:(positive_scalar*len)] <- chrs_tem
starts[1:(positive_scalar*len)] <- starts_tem
ends[1:(positive_scalar*len)] <- ends_tem
starts_coor[1:(positive_scalar*len)] <- starts_coor_tem
ends_coor[1:(positive_scalar*len)] <- ends_coor_tem
locs[1:(positive_scalar*len)] <- locs_tem


######################################################
#training
#write bed files
write.table(cbind(interaction_train, chrs, starts, ends), file = promoter_bed_file_name, sep="\t", row.names=F, col.names = F, quote = F)


rngs <- GRanges(chrs, IRanges(starts, ends))
seqs <- getSeq(Hsapiens, rngs)

str <- ""
for (i in 1:length(seqs)){
  #print(i)
  str_pre <- processN(toString(seqs[i]))
  str <- paste(">", locs[i], "\n", str_pre, sep="")    
  write(str,file = promoter_file_name, append = T)
}


#z-score
write.table(rbind(c(mean(starts_coor), sd(starts_coor)),c(mean(ends_coor), sd(ends_coor))), file = promoter_coor_file_name_stat, sep="\t", row.names = F, col.names = F)

starts_coor <- scale(starts_coor)
ends_coor <- scale(ends_coor)

#write coordinates
write.table(cbind(starts_coor, ends_coor), file = promoter_coor_file_name, sep="\t", row.names = F, col.names = F)



######################################################
#imbalanced training
#write bed files
write.table(cbind(interaction, im_chrs, im_starts, im_ends), file = im_promoter_bed_file_name, sep="\t", row.names=F, col.names = F, quote = F)


rngs <- GRanges(im_chrs, IRanges(im_starts, im_ends))
seqs <- getSeq(Hsapiens, rngs)

str <- ""
for (i in 1:length(seqs)){
  #print(i)
  str_pre <- processN(toString(seqs[i]))
  str <- paste(">", im_locs[i], "\n", str_pre, sep="")    
  write(str,file = im_promoter_file_name, append = T)
}


#z-score
write.table(rbind(c(mean(im_starts_coor), sd(im_starts_coor)),c(mean(im_ends_coor), sd(im_ends_coor))), file = im_promoter_coor_file_name_stat, sep="\t", row.names = F, col.names = F)

im_starts_coor <- scale(im_starts_coor)
im_ends_coor <- scale(im_ends_coor)

#write coordinates
write.table(cbind(im_starts_coor, im_ends_coor), file = im_promoter_coor_file_name, sep="\t", row.names = F, col.names = F)





############################################################################################################
#training
#write the label to file
inter_update <- c()
for (i in 1:length(interaction)){
  if (interaction[i]==1){
    inter_update <- c(inter_update, rep(1, positive_scalar))
  }else{
    inter_update <- c(inter_update,0)
  }
}#for i
write(inter_update, file = label_file_name, ncolumns = 1)



#imbalanced training
#write the label to file
write(interaction, file = im_label_file_name, ncolumns = 1)



