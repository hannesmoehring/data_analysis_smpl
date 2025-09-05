import textual_help as th
from tools.IMS import IntraMotionSimilarity

h3d_df = th.read_data_texts("../datasets/humanml3d/texts")
data_h3d = th.intramotion_similarity_nb(h3d_df)
