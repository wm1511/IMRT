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
	Intersection() : t(std::numeric_limits<double>::infinity()), object(nullptr)
	{
	}

	Intersection(const double t, std::shared_ptr<Object>& object) : t(t), object(object)
	{
	}

	explicit operator bool() const { return object != nullptr; }

	double t;
	std::shared_ptr<Object> object;

};