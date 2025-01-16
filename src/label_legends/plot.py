import polars as pl
from label_legends.util import ROOT
import altair


altair.renderers.enable("browser")

pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_rows(100)

scores = pl.read_csv(ROOT / "resource" / "ensemble_scores.csv").sort("recall", descending=True)
scores.head()
predictions = pl.read_csv(ROOT / "resource" / "ensemble_predictions.csv")
predictions.head()



scores.plot.bar(x="model", y="recall").show()

