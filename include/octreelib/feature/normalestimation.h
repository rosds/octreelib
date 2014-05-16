/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, University of Bonn, Computer Science Institute VI
 *  Author: Torsten Fiolka, Joerg Stueckler 04/2011
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of University of Bonn, Computer Science Institute 
 *     VI nor the names of its contributors may be used to endorse or 
 *     promote products derived from this software without specific 
 *     prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef NORMAL_ESTIMATION_OCTREE_H_
#define NORMAL_ESTIMATION_OCTREE_H_

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <octreelib/spatialaggregate/octree.h>
#include <octreelib/algorithm/downsample.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/vector_average.h>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

namespace feature
{

	class NormalEstimationValue {
	public:
		
		NormalEstimationValue()
		{
			clear();
		}

		NormalEstimationValue( const int value )
		{
			clear();
			summedSquares.Constant(value);
			summedPos.Constant(value);
		}

		void clear()
		{
			summedSquares.Zero();
			summedPos.Zero();
			pointCloudIndex = 0;
			normal.Zero();
			curvature = 0;
			stable = false;
		}

		NormalEstimationValue& operator+( const NormalEstimationValue& rhs )
		{
			summedSquares += rhs.summedSquares;
			summedPos += rhs.summedPos;
			return *this;
		}

		NormalEstimationValue& operator+=( const NormalEstimationValue& rhs )
		{
			summedSquares += rhs.summedSquares;
			summedPos += rhs.summedPos;
			return *this;
		}

		int pointCloudIndex;

		Eigen::Matrix3f summedSquares;
		Eigen::Vector3f summedPos;

		Eigen::Vector3f normal;
		float curvature;
		bool stable;

	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};

	
	template<typename CoordType, typename ValueType, typename PointType> 
	boost::shared_ptr< spatialaggregate::OcTree<CoordType, ValueType> > buildNormalEstimationOctree( const pcl::PointCloud<PointType>& cloud, const std::vector< int >& indices, int& octreeDepth, algorithm::OcTreeSamplingMap<CoordType, ValueType>& octreeSamplingMap, CoordType maxRange, CoordType minResolution, boost::shared_ptr< spatialaggregate::OcTreeNodeAllocator< CoordType, ValueType > > allocator );

	template<typename CoordType, typename ValueType, typename PointType> 
	boost::shared_ptr< spatialaggregate::OcTree<CoordType, ValueType> > buildNormalEstimationOctree( const boost::shared_ptr< const pcl::PointCloud<PointType> >& cloud, const boost::shared_ptr< const std::vector< int > >& indices, int& octreeDepth, algorithm::OcTreeSamplingMap<CoordType, ValueType>& octreeSamplingMap, CoordType maxRange, CoordType minResolution, boost::shared_ptr< spatialaggregate::OcTreeNodeAllocator< CoordType, ValueType > > allocator );
	
	template<typename CoordType, typename ValueType>
	bool calculateNormal( spatialaggregate::OcTreeNode<CoordType, ValueType>* treenode, Eigen::Vector3f& normal, int minimumPointsForNormal );

	template<typename CoordType, typename ValueType>
	void calculateNormalsOnOctreeLayer( std::vector<spatialaggregate::OcTreeNode<CoordType, ValueType>*>& layer, int minimumPointsForNormal );

	
	template<typename CoordType, typename ValueType>
	bool calculateNormalAndCurvature( spatialaggregate::OcTreeNode<CoordType, ValueType>* treenode, Eigen::Vector3f& normal, float& curvature, int minimumPointsForNormal );

	template<typename CoordType, typename ValueType>
	void calculateNormalsAndCurvaturesOnOctreeLayer( std::vector<spatialaggregate::OcTreeNode<CoordType, ValueType>*>& layer, int minimumPointsForNormal );
	
}; // namespace

#include <octreelib/feature/impl/normalestimation.hpp>

#endif /* NORMAL_ESTIMATION_OCTREE_H_ */

