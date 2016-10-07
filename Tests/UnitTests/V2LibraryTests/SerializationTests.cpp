#include "CNTKLibrary.h"
#include "Common.h"
#include <string>
#include <random>
#include <vector>


using namespace CNTK;
using namespace std;

using namespace Microsoft::MSR::CNTK;

static const size_t maxNestingDepth = 10;
static const size_t maxNestedDictSize = 10;
static const size_t maxNestedVectorSize = 100;
static const size_t maxNDShapeSize = 100;

static const size_t maxNumAxes = 10;
static const size_t maxDimSize = 15;


static size_t keyCounter = 0;
static uniform_real_distribution<double> double_dist = uniform_real_distribution<double>();
static uniform_real_distribution<float> float_dist = uniform_real_distribution<float>();

static std::wstring tempFilePath = L"serialization.tmp";

DictionaryValue CreateDictionaryValue(DictionaryValue::Type, size_t);

DictionaryValue::Type GetType()
{
    return DictionaryValue::Type(rng() % (unsigned int) DictionaryValue::Type::NDArrayView + 1);
}

void AddKeyValuePair(Dictionary& dict, size_t depth)
{
    auto type = GetType();
    while (depth >= maxNestingDepth && 
           type == DictionaryValue::Type::Vector ||
           type == DictionaryValue::Type::Dictionary)
    {
        type = GetType();
    }
    dict[L"key" + to_wstring(keyCounter++)] = CreateDictionaryValue(type, depth);
}

Dictionary CreateDictionary(size_t size, size_t depth = 0) 
{
    Dictionary dict;
    for (auto i = 0; i < size; ++i)
    {
        AddKeyValuePair(dict, depth);
    }

    return dict;
}

template <typename ElementType>
NDArrayViewPtr CreateNDArrayView(size_t numAxes, const DeviceDescriptor& device) 
{
    NDShape viewShape(numAxes);
    for (size_t i = 0; i < numAxes; ++i)
        viewShape[i] = (rng() % maxDimSize) + 1;

    return NDArrayView::RandomUniform<ElementType>(viewShape, ElementType(-4.0), ElementType(19.0), 1, device);
}

NDArrayViewPtr CreateNDArrayView()
{
    auto numAxes = (rng() % maxNumAxes) + 1;
    auto device = DeviceDescriptor::CPUDevice();
#ifndef CPUONLY
    if (rng() % 2 == 0)
    {
        device = DeviceDescriptor::GPUDevice(0);
    }
#endif

    return (rng() % 2 == 0) ? 
        CreateNDArrayView<float>(numAxes, device) : CreateNDArrayView<double>(numAxes, device);
}

DictionaryValue CreateDictionaryValue(DictionaryValue::Type type, size_t depth)
{
    switch (type)
    {
    case DictionaryValue::Type::Bool:
        return DictionaryValue(!!(rng() % 2));
    case DictionaryValue::Type::SizeT:
        return DictionaryValue(rng());
    case DictionaryValue::Type::Float:
        return DictionaryValue(float_dist(rng));
    case DictionaryValue::Type::Double:
        return DictionaryValue(double_dist(rng));
    case DictionaryValue::Type::String:
        return DictionaryValue(to_wstring(rng()));
    case DictionaryValue::Type::Axis:
        return ((rng() % 2) == 0) ? DictionaryValue(Axis(0)) : DictionaryValue(Axis(L"newDynamicAxis_" + to_wstring(rng())));
    case DictionaryValue::Type::NDShape:
    {
        size_t size = rng() % maxNDShapeSize + 1;
        NDShape shape(size);
        for (auto i = 0; i < size; i++)
        {
            shape[i] = rng();
        }
        return DictionaryValue(shape);
    }
    case DictionaryValue::Type::Vector:
    {   
        auto type = GetType();
        size_t size = rng() % maxNestedVectorSize + 1;
        vector<DictionaryValue> vector(size);
        for (auto i = 0; i < size; i++)
        {
            vector[i] = CreateDictionaryValue(type, depth + 1);
        }
        return DictionaryValue(vector);
    }
    case DictionaryValue::Type::Dictionary:
        return DictionaryValue(CreateDictionary(rng() % maxNestedDictSize  + 1, depth + 1));
    case DictionaryValue::Type::NDArrayView:
        return DictionaryValue(*(CreateNDArrayView()));
    default:
        NOT_IMPLEMENTED;
    }
}

void TestDictionarySerialization(size_t dictSize) 
{
    if ((_wunlink(tempFilePath.c_str()) != 0) && (errno != ENOENT))
        std::runtime_error("Error deleting temporary test file 'serialization.tmp'.");

    Dictionary originalDict = CreateDictionary(dictSize);
    
    {
        fstream stream;
        OpenStream(stream, tempFilePath, false);
        stream << originalDict;
        stream.flush();
    }

    Dictionary deserializedDict;

    {
        fstream stream;
        OpenStream(stream, tempFilePath, true);
        stream >> deserializedDict;
    }
    
    if (originalDict != deserializedDict)
        throw std::runtime_error("TestDictionarySerialization: original and deserialized dictionaries are not identical.");
}

template <typename ElementType>
void TestLearnerSerialization(int numParameters, const DeviceDescriptor& device) 
{
    if ((_wunlink(tempFilePath.c_str()) != 0) && (errno != ENOENT))
        std::runtime_error("Error deleting temporary test file 'serialization.tmp'.");

    NDShape shape = CreateShape(5, maxDimSize);

    unordered_set<Parameter> parameters;
    unordered_map<Parameter, NDArrayViewPtr> gradientValues;
    for (int i = 0; i < numParameters; i++)
    {
        Parameter parameter(NDArrayView::RandomUniform<ElementType>(shape, -0.5, 0.5, i, device), L"parameter_" + to_wstring(i));
        parameters.insert(parameter);
        gradientValues[parameter] = NDArrayView::RandomUniform<ElementType>(shape, -0.5, 0.5, numParameters + i, device);
    }

    auto learner1 = SGDLearner(parameters, 0.05);
    
    learner1->Update(gradientValues, 1);

    {
        auto checkpoint = learner1->Serialize();
        fstream stream;
        OpenStream(stream, tempFilePath, false);
        stream << checkpoint;
        stream.flush();
    }

    auto learner2 = SGDLearner(parameters, 0.05);

    {
        Dictionary checkpoint;
        fstream stream;
        OpenStream(stream, tempFilePath, true);
        stream >> checkpoint;
        learner2->RestoreFromCheckpoint(checkpoint);
    }

    int i = 0;
    for (auto parameter : parameters)
    {
        gradientValues[parameter] = NDArrayView::RandomUniform<ElementType>(shape, -0.5, 0.5, 2*numParameters + i, device);
        i++;
    }

    learner1->Update(gradientValues, 1);
    learner2->Update(gradientValues, 1);

     auto checkpoint1 = learner1->Serialize();
     auto checkpoint2 = learner2->Serialize();
    
    if (checkpoint1 != checkpoint2)
        throw std::runtime_error("TestLearnerSerialization: original and restored from a checkpoint learners diverge.");
}


void CheckEnumValuesNotModified() {
    // During the model and checkpoint serialization, for all enum values we save corresponding 
    // integer values. For this reason, we need to make sure that enum values never change 
    // corresponding integer values (new enum values can only be appended to the end of the value
    // list and never inserted in the middle). 

    // The following list of asserts is APPEND ONLY. DO NOT CHANGE existing assert statements.

    
    static_assert(static_cast<size_t>(DataType::Unknown) == 0 &&
                  static_cast<size_t>(DataType::Float) == 1 &&
                  static_cast<size_t>(DataType::Double) == 2, 
                  "DataType enum value was modified.");

    static_assert(static_cast<size_t>(VariableKind::Input) == 0 &&
                  static_cast<size_t>(VariableKind::Output) == 1 &&
                  static_cast<size_t>(VariableKind::Parameter) == 2 &&
                  static_cast<size_t>(VariableKind::Constant) == 3 &&
                  static_cast<size_t>(VariableKind::Placeholder) == 4, 
                  "VariableKind enum value was modified.");

    
    static_assert(static_cast<size_t>(PrimitiveOpType::Negate) == 0 &&
                  static_cast<size_t>(PrimitiveOpType::Sigmoid) == 1 &&
                  static_cast<size_t>(PrimitiveOpType::Tanh) == 2 &&
                  static_cast<size_t>(PrimitiveOpType::ReLU) == 3 &&
                  static_cast<size_t>(PrimitiveOpType::Exp) == 4 &&
                  static_cast<size_t>(PrimitiveOpType::Log) == 5 &&
                  static_cast<size_t>(PrimitiveOpType::Sqrt) == 6 &&
                  static_cast<size_t>(PrimitiveOpType::Floor) == 7 &&
                  static_cast<size_t>(PrimitiveOpType::Abs) == 8 &&
                  static_cast<size_t>(PrimitiveOpType::Reciprocal) == 9 &&
                  static_cast<size_t>(PrimitiveOpType::Softmax) == 10 &&
                  static_cast<size_t>(PrimitiveOpType::Hardmax) == 11 &&
                  static_cast<size_t>(PrimitiveOpType::TransposeAxes) == 12 &&
                  static_cast<size_t>(PrimitiveOpType::Where) == 13 &&
                  static_cast<size_t>(PrimitiveOpType::Slice) == 14 &&
                  static_cast<size_t>(PrimitiveOpType::Dropout) == 15 &&
                  static_cast<size_t>(PrimitiveOpType::Reshape) == 16 &&
                  static_cast<size_t>(PrimitiveOpType::Pooling) == 17 &&
                  static_cast<size_t>(PrimitiveOpType::SumAll) == 18 &&
                  static_cast<size_t>(PrimitiveOpType::Plus) == 19  &&
                  static_cast<size_t>(PrimitiveOpType::Minus) == 20 &&
                  static_cast<size_t>(PrimitiveOpType::ElementTimes) == 21 &&
                  static_cast<size_t>(PrimitiveOpType::Equal) == 22 &&
                  static_cast<size_t>(PrimitiveOpType::NotEqual) == 23 &&
                  static_cast<size_t>(PrimitiveOpType::Less) == 24 &&
                  static_cast<size_t>(PrimitiveOpType::LessEqual) == 25 &&
                  static_cast<size_t>(PrimitiveOpType::Greater) == 26 &&
                  static_cast<size_t>(PrimitiveOpType::GreaterEqual) == 27 &&
                  static_cast<size_t>(PrimitiveOpType::PackedIndex) == 28 &&
                  static_cast<size_t>(PrimitiveOpType::GatherPacked) == 29 &&
                  static_cast<size_t>(PrimitiveOpType::ScatterPacked) == 30 &&
                  static_cast<size_t>(PrimitiveOpType::Times) == 31 &&
                  static_cast<size_t>(PrimitiveOpType::TransposeTimes) == 32 &&
                  static_cast<size_t>(PrimitiveOpType::Convolution) == 33 &&
                  static_cast<size_t>(PrimitiveOpType::SquaredError) == 34 &&
                  static_cast<size_t>(PrimitiveOpType::CrossEntropyWithSoftmax) == 35 &&
                  static_cast<size_t>(PrimitiveOpType::ClassificationError) == 36 &&
                  static_cast<size_t>(PrimitiveOpType::PastValue) == 37 &&
                  static_cast<size_t>(PrimitiveOpType::FutureValue) == 38 &&
                  static_cast<size_t>(PrimitiveOpType::ReduceElements) == 39 &&
                  static_cast<size_t>(PrimitiveOpType::BatchNormalization) == 40 &&
                  static_cast<size_t>(PrimitiveOpType::Clip) == 41 &&
                  static_cast<size_t>(PrimitiveOpType::Select) == 42 &&
                  static_cast<size_t>(PrimitiveOpType::Splice) == 43 &&
                  static_cast<size_t>(PrimitiveOpType::Combine) == 44, 
                  "PrimitiveOpType enum value was modified.");
}


static Trainer BuildTrainer(const FunctionPtr& function, const Variable& labels, const LearningRatesPerSample& learningRateSchedule)
{
    auto trainingLoss = CNTK::CrossEntropyWithSoftmax(function, labels, L"lossFunction");
    auto prediction = CNTK::ClassificationError(function, labels, L"classificationError");
    auto learner = SGDLearner(function->Parameters(), learningRateSchedule);
   return Trainer(function, trainingLoss, prediction, { learner });
}

void TestModelSerialization(const DeviceDescriptor& device)
{
    const size_t inputDim = 2000;
    const size_t cellDim = 25;
    const size_t hiddenDim = 25;
    const size_t embeddingDim = 50;
    const size_t numOutputClasses = 5;

    auto features = InputVariable({ inputDim }, true /*isSparse*/, DataType::Float, L"features");
    auto classifierOutput = LSTMSequenceClassiferNet(features, numOutputClasses, embeddingDim, hiddenDim, cellDim, device, L"classifierOutput");

    auto labels = InputVariable({ numOutputClasses }, DataType::Float, L"labels", { Axis::DefaultBatchAxis() });

    auto minibatchSource = TextFormatMinibatchSource(L"Train.ctf", { { L"features", inputDim, true, L"x" }, { L"labels", numOutputClasses, false, L"y" } }, 0);
    auto featureStreamInfo = minibatchSource->StreamInfo(features);
    auto labelStreamInfo = minibatchSource->StreamInfo(labels);

    const size_t minibatchSize = 200;
    auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
    auto actualMBSize = minibatchData[labelStreamInfo].m_numSamples;

    LearningRatesPerSample learningRateSchedule({ { 2, 0.0005 }, { 2, 0.00025 } }, actualMBSize);

    Trainer trainer = BuildTrainer(classifierOutput, labels, learningRateSchedule);

    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);


    for (int i = 0; i < 3; ++i)
    {
        Dictionary model = Function::Save(classifierOutput);

        auto classifierOutputReloaded = Function::Load(model, device);

        std::unordered_map<Variable, Variable> replacements;
        const auto& inputs = classifierOutputReloaded->Inputs();
        for (const auto& input : inputs)
        {
            if (input.IsPlaceholder() && input.Uid() == features.Uid())
            {
                replacements[input] = features;
            }
        }

        classifierOutputReloaded->ReplacePlaceholders(replacements);

        Trainer trainerReloaded = BuildTrainer(classifierOutputReloaded, labels, learningRateSchedule);

        for (int j = 0; j < 2; ++j)
        {
            trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
            trainerReloaded.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);

            double mbLoss1 = trainer.PreviousMinibatchLossAverage();
            double mbLoss2 = trainerReloaded.PreviousMinibatchLossAverage();

            if (mbLoss1 != mbLoss2)
                throw std::runtime_error("Post checkpoint restoration training loss does not match expectation");
        }
    }
}

void SerializationTests()
{
    TestDictionarySerialization(4);
    TestDictionarySerialization(8);
    TestDictionarySerialization(16);

    TestLearnerSerialization<float>(5, DeviceDescriptor::CPUDevice());
    TestLearnerSerialization<double>(10, DeviceDescriptor::CPUDevice());

    TestModelSerialization(DeviceDescriptor::GPUDevice(0));
    TestModelSerialization(DeviceDescriptor::CPUDevice());

#ifndef CPUONLY
    TestLearnerSerialization<float>(5, DeviceDescriptor::GPUDevice(0));
    TestLearnerSerialization<double>(10, DeviceDescriptor::GPUDevice(0));;
    TestModelSerialization(DeviceDescriptor::GPUDevice(0));
#endif
    
}
