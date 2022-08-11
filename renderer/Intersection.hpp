// Copyright (c) 2022, Wiktor Merta
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "Object.hpp"

class Intersection
{
public:
	Intersection() : mT(std::numeric_limits<float>::infinity()), mObject(nullptr)
	{
	}

	Intersection(const float t, std::shared_ptr<Object>& object) : mT(t), mObject(object)
	{
	}

	explicit operator bool() const { return mObject != nullptr; }

	[[nodiscard]] float getT() const { return mT; }
	[[nodiscard]] std::shared_ptr<Object> getObject() const { return mObject; }
	void setT(const float t) { mT = t; }
	void setObject(const std::shared_ptr<Object>& object) { mObject = object; }

private:
	float mT;
	std::shared_ptr<Object> mObject;

};