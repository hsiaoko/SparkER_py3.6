import py_sparker as sparker
from pyspark import SparkContext


pth_1 = "/data/DBLP2.csv"
pth_2 = "/data/ACM.csv"
pth_gt = "/data/DBLP-ACM_perfectMapping.csv"

sc = SparkContext.getOrCreate()

profiles1 = sparker.CSVWrapper.loadProfiles(pth_1, header = True, realIDField = "id")
#Max profile id in the first dataset, used to separate the profiles in the next phases
separatorID = profiles1.map(lambda profile: profile.profileID).max()


profiles2 = sparker.CSVWrapper.loadProfiles(pth_2, header = True, realIDField = "id", startIDFrom = separatorID+1, sourceId=1)

separatorIDs = [separatorID]
#separatorIDs = []   
maxProfileID = profiles2.map(lambda profile: profile.profileID).max()
profiles = profiles1.union(profiles2)


groundtruth = sparker.CSVWrapper.loadGroundtruth(pth_gt, id1="idDBLP", id2="idACM")


realIdIds1 = sc.broadcast(profiles1.map(lambda p:(p.originalID, p.profileID)).collectAsMap())
realIdIds2 = sc.broadcast(profiles2.map(lambda p:(p.originalID, p.profileID)).collectAsMap())

def convert(gtEntry):
    if gtEntry.firstEntityID in realIdIds1.value and gtEntry.secondEntityID in realIdIds2.value:
        first = realIdIds1.value[gtEntry.firstEntityID]
        second = realIdIds2.value[gtEntry.secondEntityID]
        if (first < second):
            return (first, second)
        else:
            return (second, first)
    else:
        return (-1, -1)


newGT = sc.broadcast(set(groundtruth.map(convert).filter(lambda x: x[0] >= 0).collect()))
realIdIds1.unpersist()
realIdIds2.unpersist()

blocks = sparker.TokenBlocking.createBlocks(profiles, separatorIDs)

blocksPurged = sparker.BlockPurging.blockPurging(blocks, 1.005)

(profileBlocks, profileBlocksFiltered, blocksAfterFiltering) = sparker.BlockFiltering.blockFilteringQuick(blocksPurged, 0.8, separatorIDs)


blockIndexMap = blocksAfterFiltering.map(lambda b : (b.blockID, b.profiles)).collectAsMap()
blockIndex = sc.broadcast(blockIndexMap)
profileBlocksSizeIndex = sc.broadcast(profileBlocksFiltered.map(lambda pb : (pb.profileID, len(pb.blocks))).collectAsMap())

entropies = None

entropiesMap = blocks.map(lambda b : (b.blockID, b.entropy)).collectAsMap()
entropies = sc.broadcast(entropiesMap)

use_entropy = True

results = sparker.WNP.wnp(profileBlocksFiltered,
                          blockIndex,
                          maxProfileID,
                          separatorIDs,
                          newGT,
                          sparker.ThresholdTypes.AVG,#Threshold type
                          sparker.WeightTypes.CBS,#Weighting schema
                          profileBlocksSizeIndex,
                          use_entropy,
                          entropies,
                          2.0,#Blast c parameter
                          sparker.ComparisonTypes.OR#Pruning strategy
                         )
match_found = float(results.map(lambda x: x[1]).sum())
num_edges = results.map(lambda x: x[0]).sum()
candidate_set = results.flatMap(lambda x: x[2])

print("#############")
print("match_found", match_found)
pc = match_found / len(newGT.value)
pq = match_found / num_edges
print("Recall: "+str(pc)+", Precision: "+str(pq))
print("#############")
