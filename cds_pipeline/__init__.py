"""
cds_pipeline — Clean, bug-fixed CDS inference package.

Usage:
    from cds_pipeline.pipeline import CDSPipeline
    pipe = CDSPipeline(s3_version="v3")
    ranked = pipe.rank(question, candidates, path)

Or run evaluation directly:
    python -m cds_pipeline.evaluate --s3 v3
"""
