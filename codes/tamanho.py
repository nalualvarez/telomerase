# from ushuffle import shuffle, Shuffler
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from statistics import median, quantiles

# filename = './animals_asu_database.fasta'
filename = './iniciallimpo.fasta'
# filename = './negativos1limpo.fasta'

sequences = [i for i in SeqIO.parse(filename,'fasta')]

print(len(sequences))

sequence_a = sequences[0]
seqSize = []




for seq in sequences:
    seqSize.append(len(str(seq.seq)))

total = 0
minSize = 0
maxSize = 0

for size in seqSize:
    total = total + size

    if(size < minSize or minSize == 0):
        minSize = size

    if(size > maxSize or maxSize == 0):
        maxSize = size

average = total / len(seqSize)
medianValue = median(seqSize)

print(seqSize)
print(f"Total: {total}")
print(f"Média: {average}")
print(f"Tamanho mínimo: {minSize}")
print(f"Tamanho máximo: {maxSize}")
print(f"Mediana: {medianValue}")
print(quantiles(seqSize, n=4))
