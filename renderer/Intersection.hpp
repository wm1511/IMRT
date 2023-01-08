#pragma once
#include "Object.hpp"

class Intersection
{
public:
	Intersection() : t_(std::numeric_limits<double>::infinity()), object_(nullptr)
	{
	}

	Intersection(const double t, std::shared_ptr<Object>& object) : t_(t), object_(object)
	{
	}

	explicit operator bool() const { return object_ != nullptr; }

	double t_;
	std::shared_ptr<Object> object_;

};