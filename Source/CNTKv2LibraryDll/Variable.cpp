//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Serialization.h"
#include "Function.h"

namespace CNTK
{
    Variable::Variable(const FunctionPtr& function)
        : Variable(function->Output())
    {
    }

    FunctionPtr Variable::Owner() const 
    {
        if (m_dataFields->m_ownerFunction != nullptr)
            return m_dataFields->m_ownerFunction->shared_from_this();
        else
            return nullptr;
    }

    Variable::operator FunctionPtr() const
    {
        auto varOwner = Owner();
        if (varOwner)
            return CompositeFunction::Create(varOwner, varOwner->Name());
        else
            return Internal::Combine({ *this });
    }

    /*static*/ Parameter Parameter::UniformInitParameter(const NDShape& shape, DataType type, double range, unsigned long seed, const DeviceDescriptor& device, const std::wstring& name)
    {
        switch (type)
        {
        case DataType::Float:
            return Parameter(NDArrayView::RandomUniform<float>(shape, -range, range, seed, device), name);
        case DataType::Double:
            return Parameter(NDArrayView::RandomUniform<double>(shape, -range, range, seed, device), name);
        default:
            InvalidArgument("Parameter construction: Unsupported DataType %s", DataTypeName(type));
        }
    }

    /*static*/ Parameter Parameter::NormalInitParameter(const NDShape& shape, DataType type, double stdDev, unsigned long seed, const DeviceDescriptor& device, const std::wstring& name)
    {
        switch (type)
        {
        case DataType::Float:
            return Parameter(NDArrayView::RandomNormal<float>(shape, 0, stdDev, seed, device), name);
        case DataType::Double:
            return Parameter(NDArrayView::RandomNormal<double>(shape, 0, stdDev, seed, device), name);
        default:
            InvalidArgument("Parameter construction: Unsupported DataType %s", DataTypeName(type));
        }
    }

    static const std::wstring s_variableTypeValue = L"Variable";

    /*virtual*/ Dictionary Variable::Serialize() const
    {
        if (IsOutput())
        {
            LogicError("Output variables cannot be saved");
        }
        Dictionary dict;

        dict[versionKey] = CurrentVersion();
        dict[typeKey] = s_variableTypeValue;
        dict[uidKey] = Uid();
        dict[kindKey] = static_cast<size_t>(Kind());
        dict[dataTypeKey] = static_cast<size_t>(GetDataType());
        const auto& dynamicAxis = DynamicAxes();
        vector<DictionaryValue> dictionaryValueVector; 
        dictionaryValueVector.reserve(dynamicAxis.size());
        for (const auto& axis : dynamicAxis)
        {
            dictionaryValueVector.push_back(axis);
        }
        dict[dynamicAxisKey] = dictionaryValueVector;
        dict[isSparseKey] = IsSparse();
        dict[nameKey] = Name();
        dict[needsGradientKey] = NeedsGradient();
        dict[shapeKey] = Shape();
        if (IsParameter() || IsConstant())
        {
            // TODO: add a dictionary value constructor with an rvalue parameter.
            dict[valueKey] = DictionaryValue(*(Value().get()));
        }
        return dict;
    }

    /*static*/ Variable Variable::Load(const Dictionary& dict, const CNTK::DeviceDescriptor& device)
    {
        static const vector<std::wstring> s_requiredDictionaryKeys = { typeKey, uidKey, kindKey, dataTypeKey, dynamicAxisKey, isSparseKey, nameKey, needsGradientKey, shapeKey };

        size_t version = ValidateDictionary<Variable>(dict, s_requiredDictionaryKeys, s_modelVersion);

        ValidateType<Variable>(dict, s_variableTypeValue, s_modelVersion);

        const auto& uid = dict[uidKey].Value<std::wstring>();

        VariableKind kind = VariableKind(dict[kindKey].Value<std::size_t>());
        if (kind != VariableKind::Constant &&
            kind != VariableKind::Input &&
            kind != VariableKind::Parameter &&
            kind != VariableKind::Placeholder)
        {
            LogicError("Unexpected variable '%ls':'%zu' "
                        "(%s).", kindKey, kind, GetVersionsString<Variable>(s_modelVersion, version));
        }
        
        DataType dataType = DataType(dict[dataTypeKey].Value<std::size_t>());
        if (dataType != DataType::Unknown &&
            dataType != DataType::Float &&
            dataType != DataType::Double)
        {
            LogicError("Unexpected variable '%ls':'%zu' "
                        "(%s).", dataTypeKey, dataType, GetVersionsString<Variable>(s_modelVersion, version));
        }
        
        const vector<DictionaryValue>& dictionaryValueVector = dict[dynamicAxisKey].Value<vector<DictionaryValue>>();
        vector<Axis> dynamicAxis;
        dynamicAxis.reserve(dictionaryValueVector.size());
        for (const auto& dictionaryValue : dictionaryValueVector)
        {
            dynamicAxis.push_back(dictionaryValue.Value<Axis>());
        }

        bool isSparse = dict[isSparseKey].Value<bool>();
        const auto& name = dict[nameKey].Value<std::wstring>();
        bool needsGradient = dict[needsGradientKey].Value<bool>();
        const auto& shape = dict[shapeKey].Value<NDShape>();

        if (kind == VariableKind::Constant || kind == VariableKind::Parameter)
        {
            auto& value = dict[valueKey].Value<NDArrayView>();

            // TODO: this copying here is redundant, value should be moved from the dictionary to the variable.
            // Also, the correct device should be used upfront when deserializing NDArrayView.
            Variable var(shape, kind, dataType, nullptr, value.DeepClone(device, kind == VariableKind::Constant), needsGradient, dynamicAxis, isSparse, name, uid);
            if (var.IsParameter())
            {
                return Parameter(var);
            }
            else
            {
                return Constant(var);
            }
        }

        return Variable(shape, kind, dataType, nullptr, nullptr, needsGradient, dynamicAxis, isSparse, name, uid);
    }
}
