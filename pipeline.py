# pipeline.py
import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ScriptProcessor

# ── Session setup ─────────────────────────────────────────────────────────────
region     = boto3.Session().region_name
role       = sagemaker.get_execution_role()
sess       = PipelineSession()

BUCKET     = "insurance-pricing-engine-225635015471-us-east-2"
PREFIX     = "frequency-model"
BASE_URI   = f"s3://{BUCKET}"

print(f"Region : {region}")
print(f"Role   : {role}")
print(f"Bucket : {BUCKET}")

# ── Step 1: Preprocessing ─────────────────────────────────────────────────────
# SKLearnProcessor gives us a managed container with sklearn pre-installed
# ml.m5.xlarge = 2 vCPU, 8GB RAM — plenty for 30MB dataset
sklearn_processor = SKLearnProcessor(
    framework_version  = "1.2-1",
    instance_type      = "ml.m5.xlarge",
    instance_count     = 1,
    role               = role,
    sagemaker_session  = sess,
)

step_process = ProcessingStep(
    name = "FrequencyPreprocessing",
    processor = sklearn_processor,
    code = "scripts/preprocess.py",
    inputs = [
        ProcessingInput(
            source           = f"{BASE_URI}/data/raw/freMTPL2/freMTPL2freq.csv",
            destination      = "/opt/ml/processing/input",
        )
    ],
    outputs = [
        ProcessingOutput(
            output_name = "train",
            source      = "/opt/ml/processing/train",
            destination = f"{BASE_URI}/data/processed/train",
        ),
        ProcessingOutput(
            output_name = "val",
            source      = "/opt/ml/processing/val",
            destination = f"{BASE_URI}/data/processed/val",
        ),
        ProcessingOutput(
            output_name = "test",
            source      = "/opt/ml/processing/test",
            destination = f"{BASE_URI}/data/processed/test",
        ),
    ],
)

# ── Step 2: Training ──────────────────────────────────────────────────────────
# We use a generic sklearn container for training too
# The model artifact (model.tar.gz) gets saved to S3 automatically
sklearn_estimator = Estimator(
    image_uri          = sagemaker.image_uris.retrieve(
                            "sklearn",
                            region,
                            version="1.2-1",
                         ),
    instance_type      = "ml.m5.xlarge",
    instance_count     = 1,
    role               = role,
    sagemaker_session  = sess,
    entry_point        = "scripts/train.py",
    dependencies       = ["requirements.txt"],  # ← this line is new
    hyperparameters    = {
        "n-estimators"     : 300,
        "max-depth"        : 8,
        "learning-rate"    : 0.05,
        "subsample"        : 0.8,
        "colsample-bytree" : 0.8,
    },
    output_path        = f"{BASE_URI}/models/frequency/artifacts",
)

step_train = TrainingStep(
    name      = "FrequencyModelTraining",
    estimator = sklearn_estimator,
    inputs    = {
        "train": TrainingInput(
            s3_data    = step_process.properties.ProcessingOutputConfig.Outputs[
                "train"].S3Output.S3Uri,
            content_type = "text/csv",
        ),
        "val": TrainingInput(
            s3_data    = step_process.properties.ProcessingOutputConfig.Outputs[
                "val"].S3Output.S3Uri,
            content_type = "text/csv",
        ),
    },
)

# ── Step 3: Evaluation ────────────────────────────────────────────────────────
eval_processor = SKLearnProcessor(
    framework_version = "1.2-1",
    instance_type     = "ml.m5.xlarge",
    instance_count    = 1,
    role              = role,
    sagemaker_session = sess,
    env               = {"EXTRA_PIP_PACKAGES": "xgboost==1.7.6 joblib==1.2.0"},
)

step_evaluate = ProcessingStep(
    name      = "FrequencyModelEvaluation",
    processor = eval_processor,
    code      = "scripts/evaluate.py",
    inputs    = [
        ProcessingInput(
            source      = step_train.properties.ModelArtifacts.S3ModelArtifacts,
            destination = "/opt/ml/processing/model",
        ),
        ProcessingInput(
            source      = step_process.properties.ProcessingOutputConfig.Outputs[
                "test"].S3Output.S3Uri,
            destination = "/opt/ml/processing/test",
        ),
    ],
    outputs   = [
        ProcessingOutput(
            output_name = "evaluation",
            source      = "/opt/ml/processing/evaluation",
            destination = f"{BASE_URI}/models/frequency/evaluation",
        ),
    ],
)

# ── Wire steps into a Pipeline ────────────────────────────────────────────────
pipeline = Pipeline(
    name   = "InsuranceFrequencyPipeline",
    steps  = [step_process, step_train, step_evaluate],
    sagemaker_session = sess,
)

# ── Create / update the pipeline in AWS ──────────────────────────────────────
print("\nUpserting pipeline...")
pipeline.upsert(role_arn=role)
print("✓ Pipeline registered in SageMaker")

# ── Start a pipeline run ──────────────────────────────────────────────────────
execution = pipeline.start()
print(f"\n✓ Pipeline execution started")
print(f"  Execution ARN: {execution.arn}")
print(f"\nMonitor progress in SageMaker Studio:")
print(f"  Left sidebar → Pipelines → InsuranceFrequencyPipeline")