#pragma once

class IDrawable
{
public:
	virtual ~IDrawable() = default;
	virtual void init() = 0;
	virtual void draw() = 0;
};