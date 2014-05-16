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


template<typename CoordType, typename ValueType, typename PointType>
boost::shared_ptr< spatialaggregate::OcTree<CoordType, ValueType> > feature::buildNormalEstimationOctree( const pcl::PointCloud<PointType>& cloud, const std::vector< int >& indices, int& octreeDepth, algorithm::OcTreeSamplingMap<CoordType, ValueType>& octreeSamplingMap, CoordType maxRange, CoordType minResolution, boost::shared_ptr< spatialaggregate::OcTreeNodeAllocator< CoordType, ValueType > > allocator ) {
	
	boost::shared_ptr< spatialaggregate::OcTree<CoordType, ValueType> > octree = boost::make_shared< spatialaggregate::OcTree<CoordType, ValueType> >( Eigen::Matrix< float, 4, 1 >( 0.f, 0.f, 0.f ), minResolution, maxRange, allocator );

	octreeDepth = 0;

	for( unsigned int i = 0; i < indices.size(); i++ ) {

		const PointType& p = cloud.points[indices[i]];
		const float x = p.x;
		const float y = p.y;
		const float z = p.z;

		
		if( isnan(x) || isinf(x) )
			continue;

		if( isnan(y) || isinf(y) )
			continue;

		if( isnan(z) || isinf(z) )
			continue;	
		
		ValueType value;
		value.summedSquares(0, 0) = x*x;
		value.summedSquares(0, 1) = value.summedSquares(1, 0) = x*y;
		value.summedSquares(0, 2) = value.summedSquares(2, 0) = x*z;
		value.summedSquares(1, 1) = y*y;
		value.summedSquares(1, 2) = value.summedSquares(2, 1) = y*z;
		value.summedSquares(2, 2) = z*z;

		value.summedPos(0) = x;
		value.summedPos(1) = y;
		value.summedPos(2) = z;
		value.pointCloudIndex = indices[i];

//		spatialaggregate::OcTreeNode<CoordType, ValueType>* n = octree->addPoint( p.getVectorMap4f(),  );

	}

	octreeSamplingMap.clear();
	octreeSamplingMap = algorithm::downsampleOcTree(*octree, false, octree->max_depth_, cloud.points.size());
	
	return octree;
	
}


template<typename CoordType, typename ValueType, typename PointType>
boost::shared_ptr< spatialaggregate::OcTree<CoordType, ValueType> > feature::buildNormalEstimationOctree( const boost::shared_ptr< const pcl::PointCloud<PointType> >& cloud, const boost::shared_ptr< const std::vector< int > >& inputIndices, int& octreeDepth, algorithm::OcTreeSamplingMap<CoordType, ValueType>& octreeSamplingMap, CoordType maxRange, CoordType minResolution, boost::shared_ptr< spatialaggregate::OcTreeNodeAllocator< CoordType, ValueType > > allocator ) {
	
	if( !inputIndices ) {
		std::vector< int > indices( cloud->points.size() );
		for( unsigned int i = 0; i < indices.size(); i++ )
			indices[i] = i;
		return feature::buildNormalEstimationOctree( *cloud, indices, octreeDepth, octreeSamplingMap, maxRange, 	minResolution, allocator );
	}
	else
		return feature::buildNormalEstimationOctree( *cloud, *inputIndices, octreeDepth, octreeSamplingMap, maxRange, 	minResolution, allocator );

	
}




//! Gets the normal of a given integral value and a point count
template<typename CoordType, typename ValueType>
bool feature::calculateNormal( spatialaggregate::OcTreeNode<CoordType, ValueType>* treenode, Eigen::Vector3f& normal, int minimumPointsForNormal ) {
	
	int count = treenode->numPoints;
	if( count < minimumPointsForNormal ) {
		return false;
	}
	
	Eigen::Matrix3f summedSquares(treenode->value.summedSquares);
	Eigen::Vector3f summedPos(treenode->value.summedPos);

	const float invCount = 1.f / ((float)treenode->numPoints);
	
	summedSquares *= invCount;
	summedPos *= invCount;

	summedSquares -= summedPos * summedPos.transpose();

	Eigen::Matrix<float, 3, 1> eigen_values;
	Eigen::Matrix<float, 3, 3> eigen_vectors;
	pcl::eigen33(summedSquares, eigen_vectors, eigen_values);

	normal = eigen_vectors.col(0);
	
	if( summedPos.dot( normal ) > 0.f )
		normal *= -1.f;
	
	return true;
}


//! Determines normal and curvature from a node's integral value and point count
template<typename CoordType, typename ValueType>
bool feature::calculateNormalAndCurvature( spatialaggregate::OcTreeNode<CoordType, ValueType>* treenode, Eigen::Vector3f& normal, float& curvature, int minimumPointsForNormal ) {
	
	int count = treenode->numPoints;
	if( count < minimumPointsForNormal ) {
		return false;
	}
	
	Eigen::Matrix3f summedSquares(treenode->value.summedSquares);
	Eigen::Vector3f summedPos(treenode->value.summedPos);

	const float invCount = 1.f / ((float)treenode->numPoints);
	
	summedSquares *= invCount;
	summedPos *= invCount;

	summedSquares -= summedPos * summedPos.transpose();

	Eigen::Matrix<float, 3, 1> eigen_values;
	Eigen::Matrix<float, 3, 3> eigen_vectors;
	pcl::eigen33(summedSquares, eigen_vectors, eigen_values);

	normal = eigen_vectors.col(0);
	
	if( summedPos.dot( normal ) > 0.f )
		normal *= -1.f;
	
	float sumEigVals = eigen_values(0) + eigen_values(1) + eigen_values(2);
	if( sumEigVals > 1e-10f )
		curvature = eigen_values(0) / sumEigVals;
	else
		curvature = 0.f;
	
	return true;
}


template<typename CoordType, typename ValueType>
void feature::calculateNormalsOnOctreeLayer( std::vector<spatialaggregate::OcTreeNode<CoordType, ValueType>*>& layer, int minimumPointsForNormal ) {  
	
	for( unsigned int i=0; i<layer.size(); i++ ) {
		layer[i]->value.stable = feature::calculateNormal< CoordType, ValueType>( layer[i], layer[i]->value.normal, minimumPointsForNormal);
	}
	
}


template<typename CoordType, typename ValueType>
void feature::calculateNormalsAndCurvaturesOnOctreeLayer( std::vector<spatialaggregate::OcTreeNode<CoordType, ValueType>*>& layer, int minimumPointsForNormal ) {  
	
	for( unsigned int i=0; i<layer.size(); i++ ) {
		layer[i]->value.stable = feature::calculateNormalAndCurvature< CoordType, ValueType>( layer[i], layer[i]->value.normal, layer[i]->value.curvature, minimumPointsForNormal);
	}
	
}

