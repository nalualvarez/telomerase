# from ushuffle import shuffle, Shuffler
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import random as rn

# filename = './animals_asu_database.fasta'
# filename = []

# for i in range(1, 25):
#     filename.append('./hs' + str(i) + '.fasta')

# print(filename)
# filename = ['./hs1.fasta', './hs2.fasta', './hs3.fasta', './hs4.fasta', './hs5.fasta', './hs6.fasta', './hs7.fasta', './hs8.fasta']
filenamei = './patellavulgata.fasta'
filenameo = []

for i in range(1, 11):
    filenameo.append('./pv' + str(i) + '.fasta')


sequences = [j for j in SeqIO.parse(filenamei,'fasta')]
print(len(sequences))

seqA = sequences[0]
seqAstr = str(sequences[0].seq)

index = 0

tammaxseq = len(seqAstr)

for i in range(0,10):
    seqRead = []

    for k in range(0, 30):
        size = rn.randint(370, 510)

        index = rn.randint(0,tammaxseq-520)

        seq = seqAstr[index:index+size]

        seqRead.append(SeqRecord(Seq(seq), id=seqA.id, name=seqA.name, description=seqA.description))

    print(seqRead)
    print(len(seqRead))
    SeqIO.write(seqRead, filenameo[i], "fasta")
