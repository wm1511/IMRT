#pragma once

class IDrawable
{
public:
	virtual ~IDrawable() = default;
	virtual void draw() = 0;
};