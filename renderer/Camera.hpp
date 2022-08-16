#pragma once

#include "Ray.hpp"

#include "glm/gtc/random.hpp"

class Camera
{
public:
	Camera(glm::dvec3 lookOrigin, glm::dvec3 lookTarget, float vfov, double aspectRatio, double aperture, double focusDistance)
	{
		double viewportHeight = 2.0 * static_cast<double>(glm::tan(vfov / 2));
		double viewportWidth = viewportHeight * aspectRatio;

		glm::dvec3 cameraDirection = normalize(lookOrigin - lookTarget);
		u = normalize(cross({0.0, -1.0, 0.0}, cameraDirection));
		v = cross(cameraDirection, u);

		mOrigin = lookOrigin;
		mHorizontalMap = focusDistance * viewportWidth * u;
		mVerticalMap = focusDistance * viewportHeight * v;
		mStart = mOrigin - mHorizontalMap / 2.0 - mVerticalMap / 2.0 - focusDistance * cameraDirection;
		mLensRadius = aperture / 2.0;
	}

	[[nodiscard]] Ray castRay(const double x, const double y) const
	{
		const glm::dvec3 randomOnLens = mLensRadius * normalize(
			glm::dvec3(glm::linearRand(-1.0, 1.0), glm::linearRand(-1.0, 1.0), 0.0));
		const glm::dvec3 offset = u * randomOnLens.x + v * randomOnLens.y;
		return Ray(mOrigin + offset, mStart + x * mHorizontalMap + y * mVerticalMap - mOrigin - offset);
	}

private:
	glm::dvec3 mOrigin{};
	glm::dvec3 mStart{};
	glm::dvec3 mHorizontalMap{};
	glm::dvec3 mVerticalMap{};
	glm::dvec3 u{}, v{};
	double mLensRadius{};

};