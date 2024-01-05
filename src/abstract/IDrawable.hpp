// Copyright Wiktor Merta 2023
#pragma once

// Abstract class representing interface being drawn currently
class IDrawable
{
public:
	virtual ~IDrawable() = default;
	virtual void draw() = 0;
};