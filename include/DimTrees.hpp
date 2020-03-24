#ifndef DOXYGEN_SHOULD_SKIP_THIS
/**
 * @date      08/08/2019
 * @author    Christos Tsalidis
 * @author    Yiorgos Lourakis
 * @author    George Lykoudis
 * @copyright 2019 Neurocom All Rights Reserved.
 */
#endif  // DOXYGEN_SHOULD_SKIP_THIS
/********************************************************************/
/**
* @file      DimTrees.hpp
* @details
* Implementation of Dimension Trees functionality.
********************************************************************/
#ifndef DIM_TREES_HPP
#define DIM_TREES_HPP

#include "PARTENSOR_basic.hpp"
#include "TensorOperations.hpp"
#include "Constants.hpp"

namespace partensor
{
  struct FactorDimTree;

  /**
   * Interface for each Node of the dimension Tree. Contains Tensor order 
   * and virtual methods for static and dynamic elements.
   */
  struct  I_TnsNode
  {
    using DataType = DefaultDataType;
    /**
     * Tensor Order
     */
    std::size_t   TnsSize;
    /**
     * Initialize tensor order.
     * @param _TnsSize [in] Tensor Order.
     */
    I_TnsNode(std::size_t _TnsSize) : TnsSize(_TnsSize)
    {  }
    /**
     * A pure virtual member.
     */
    virtual I_TnsNode *Parent() = 0;
    /**
     * A pure virtual member.
     */
    virtual I_TnsNode *Left()   = 0;
    /**
     * A pure virtual member.
     */
    virtual I_TnsNode *Right()  = 0;

    /**
     * A pure virtual member.
     */
    virtual bool Updated()       = 0;
    /**
     * A pure virtual member.
     */
    virtual void SetOutdated()   = 0;

    /**
     * A pure virtual member.
     */
    virtual int                        Key()      = 0;

    /**
     * A pure virtual member.
     */
    virtual Tensor<2>                 &Gramian()  = 0;

    /**
     * A pure virtual member.
     */
    virtual void                      *TnsDims()  = 0;

    /**
    * A pure virtual member.
    */
    virtual void                      *LabelSet() = 0;

    /**
     * A pure virtual member.
     */
    virtual void                      *DeltaSet() = 0;

    /**
     * A pure virtual member.
     */
    virtual void                      *TensorX()  = 0;

    /**
     * A pure virtual member.
     * @param key [in] Searching key value.
     */
    virtual I_TnsNode *SearchKey  (int const key                                             ) = 0;

    /**
     * A pure virtual member.
     * @param num_factors [in]     Number of Factors to update.
     * @param id          [in]     Indexing for Factor to update.
     * @param factors     [in,out] Pointer to the array factors of FactorDimTree type.
     */
    virtual void       UpdateTree(int const num_factors, int const id, FactorDimTree *factors) = 0;

  };

  /**
   * Implementation of the @c I_TnsNode interface.
   * @tparam _TnsSize Tensor order
   */
  template <std::size_t _TnsSize>
  struct TnsNode : I_TnsNode
  {
    static constexpr std::size_t TnsSize = _TnsSize; /**< Tensor Order. */
    static constexpr bool        IsNull  = false;    /**< Checks if TnsSize_ is greater than zero. */

    using DataType     = I_TnsNode::DataType;
    using Tensor_Type  = Tensor<static_cast<int>(TnsSize)>;
    using Hessian_Type = Tensor<2>;

    Tensor_Type      mTnsX;          /**< Tensor of TnsNode. */
    Hessian_Type     mGramian;       /**< Gramian Matrix. */
    int              mKey;           /**< Used for mapping Factors with leafs. */
    bool             mUpdated;       /**< Checks if node has updated data. */

    /**
     * Initialize tensor order. Set @c mKey to zero, and @c mUpdated to false.
     */
    TnsNode() : I_TnsNode(TnsSize), mKey(0), mUpdated(false)
    {  }

    /**
     * @returns If the @c ExprNode has updated data then 
     *          @c true is returned, otherwise @c false.
     */
    bool Updated() override
    {
      return mUpdated;
    }

    /**
     * Set the @c TnsNode and its children as outdated.
     */
    void SetOutdated() override
    {
      if (mUpdated)
      {
        if ( !Left() && !Right() )
        {
          mUpdated = false;
        }
        else
        {
          Left()->SetOutdated();
          Right()->SetOutdated();
          mUpdated = false;
        }
      }
    }

    /**
     * @returns The @c mKey member variable of @c TnsNode.
     */
    int Key() override
    {
      return mKey;
    }

    /**
     * @returns The @c mGramian member variable of @c TnsNode.
     */
    Hessian_Type &Gramian() override
    {
      return mGramian;
    }

    /**
     * @returns The @c mTnsX member variable of @c TnsNode.
     *          No type is included, in order to specify the 
     *          @c TnsSize of @c mTnsX explicitly.
     */
    void *TensorX() override
    {
      return &mTnsX;
    }

  };

  /**
   * @c TnsNode with @c TnsSize set as zero. Used for leaf nodes.
   */
  template <>
  struct TnsNode<0> : public I_TnsNode
  {
    static constexpr std::size_t  TnsSize = 0;       /**< TnsSize set as zero. */
    static constexpr bool         IsNull  = true;    /**< Indicates that there are no data in this TnsNode. */

    using DataType    = I_TnsNode::DataType;
    using Tensor_Type = Tensor<0>;   /**< Scalar value.*/

    /**
     * Initialize tensor order to zero.
     */
    template <typename N>
    TnsNode(N par = nullptr) : I_TnsNode(TnsSize)
    {  (void) par; }
    
    /** @returns If called from zero- @c TnsNode , there is no Parent @c TnsNode so it returns @c nullptr. */
    I_TnsNode *Parent()  override { return nullptr; }  
    /** @returns If called from zero- @c TnsNode , there is no Left Child @c TnsNode so it returns @c nullptr. */
    I_TnsNode *Left  ()  override { return nullptr; }  
    /** @returns If called from zero- @c TnsNode , there is no Right Child @c TnsNode so it returns @c nullptr. */
    I_TnsNode *Right ()  override { return nullptr; }  

    void       SetOutdated() override  { throw std::runtime_error("SetOutdated()");  }  /**< If it is called throws a runtime error. */
    bool       Updated()     override  { throw std::runtime_error("Updated()");      }  /**< If it is called throws a runtime error. */
    int        Key()         override  { throw std::runtime_error("Key()");          }  /**< If it is called throws a runtime error. */
    void      *TnsDims()     override  { throw std::runtime_error("TnsDims()");      }  /**< If it is called throws a runtime error. */
    void      *LabelSet()    override  { throw std::runtime_error("LabelSet()");     }  /**< If it is called throws a runtime error. */
    void      *DeltaSet()    override  { throw std::runtime_error("DeltaSet()");     }  /**< If it is called throws a runtime error. */
    Tensor<2> &Gramian()     override  { throw std::runtime_error("Gramian()");      }  /**< If it is called throws a runtime error. */
    void      *TensorX()     override  { throw std::runtime_error("TensorX()");      }  /**< If it is called throws a runtime error. */

    /**
     * If called from zero- @c TnsNode then throws a @c runtime_error.
     * @param num_factors [in]     Number of Factors to update.
     * @param id          [in]     Indexing for Factor to update.
     * @param factors     [in,out] Pointer to the array factors of FactorDimTree type.
     */
    void        UpdateTree(int const, int const, FactorDimTree *) override {  throw std::runtime_error("UpdateTree()");   }  /**< If it is called throws a runtime error. */
    /**
     * If called from zero- @c TnsNode then throws a @c runtime_error.
     * @param key [in] Searching key value.
     */
    I_TnsNode  *SearchKey (int const                            ) override {  throw std::runtime_error("SearchKey()");    }  /**< If it is called throws a runtime error. */
  };

  using  NullTensorType = TnsNode<0>::Tensor_Type;

  /**
   * Configuration for @c TnsNode. Representation of the tree struture.
   * @tparam _LabelSetSize    Size of the LabelSet.
   * @tparam _ParLabelSetSize Size of the LabelSet of the parent node.
   * @tparam _RootSize        Size of the LabelSet of the root node.
   */
  template <std::size_t _LabelSetSize, std::size_t _ParLabelSetSize, std::size_t _RootSize>
  struct ExprNode : public TnsNode<_LabelSetSize == _RootSize ? _LabelSetSize : _LabelSetSize + 1>
  {
    using DataType = typename I_TnsNode::DataType;

    //static_assert(LabelSetSize != 0, "Error in expansion!");
    static constexpr std::size_t LabelSetSize    = _LabelSetSize;       /**< Size of the LabelSet. */
    static constexpr std::size_t ParLabelSetSize = _ParLabelSetSize;    /**< Size of the LabelSet of the parent node. */
    static constexpr std::size_t RootSize        = _RootSize;           /**< Size of the LabelSet of the root node. */

    static constexpr bool IsRoot       = LabelSetSize == RootSize;      /**< Checks if node is the root node. */
    static constexpr bool IsFirstChild = ParLabelSetSize  == RootSize;  /**< Checks if node is a child of the the root node. */
    static constexpr bool IsLeaf       = LabelSetSize == 1;             /**< Checks if node is a leaf node. */

    static constexpr std::size_t BrotherLabelSetSize = IsRoot ? 0 : ParLabelSetSize - LabelSetSize;                         /**< Size of the LabelSet of the brother node. */
    static constexpr std::size_t TnsSize             = LabelSetSize == RootSize ? LabelSetSize : LabelSetSize + 1;          /**< Tensor order. */
    static constexpr std::size_t ParTnsSize          = IsRoot ? 0 : (IsFirstChild ? ParLabelSetSize : ParLabelSetSize+1);   /**< Tensor order of the parent node. */
    static constexpr std::size_t DIM_HALF_SIZE       = (1+LabelSetSize)/2;                                                  /**< The last index of the left child. */
    static constexpr std::size_t DIM_LEFT_SIZE       = IsLeaf ? 0 : DIM_HALF_SIZE;                                          /**< Size of the LabelSet of the left child node. */
    static constexpr std::size_t DIM_RIGHT_SIZE      = IsLeaf ? 0 : LabelSetSize - DIM_LEFT_SIZE;                           /**< Size of the LabelSet of the right child node. */

    std::array<int,TnsSize>             mTnsDims;   /**< Array of size TnsSize, with the size of every dimension of the Tensor mTnsX. */
    std::array<int,LabelSetSize>        mLabelSet;  /**< Array of size LabelSetSize, with set of indices used for identification of TnsNode. */
    std::array<int,BrotherLabelSetSize> mDeltaSet;  /**< Array of size BrotherLabelSetSize, with set of indices used for identification of the brother of TnsNode. */

    using  Tns_Node_Type        = TnsNode<TnsSize>;
    using  Parent_Tns_Node_Type = TnsNode<ParTnsSize>;
    using  Node_Type            = ExprNode<LabelSetSize,ParLabelSetSize,RootSize>;
    using  Left_Node_Type       = std::conditional_t<IsLeaf, TnsNode<0>, ExprNode<DIM_LEFT_SIZE,LabelSetSize,RootSize>>;
    using  Right_Node_Type      = std::conditional_t<IsLeaf, TnsNode<0>, ExprNode<DIM_RIGHT_SIZE,LabelSetSize,RootSize>>;

    // const int             mParLabelSetSize;
    // const int             mParParLabelSetSize;
    // const std::ptrdiff_t  mParOffset;

    using Tensor_Type        = typename Tns_Node_Type::Tensor_Type;
    using Hessian_Type       = typename Tns_Node_Type::Hessian_Type;
    using Parent_Tensor_Type = typename Parent_Tns_Node_Type::Tensor_Type;

    using Tns_Node_Type::mTnsX;
    using Tns_Node_Type::mGramian;
    using Tns_Node_Type::mKey;
    using Tns_Node_Type::mUpdated;

    I_TnsNode               *parent;       /**< Pointer for the parent node. */
    Left_Node_Type           left;         /**< Left child node. */
    Right_Node_Type          right;        /**< Right child node. */

    /**
     * Default Constructor
     */
    ExprNode() : TnsNode<TnsSize>(),
                 // mParLabelSetSize(0),
                 // mParParLabelSetSize(0),
                 // mParOffset(0),
                 parent(nullptr),
                 left(this),
                 right(this)
    {
      static_assert(IsRoot,"Wrong expansion!");
    }

    template <std::size_t _ParLabelSetSize2, std::size_t _ParParLabelSetSize, std::size_t _RootSize2>
    friend struct ExprNode;

  protected:
    /**
     * Protected Constructor.
     */
    template <std::size_t _ParLabelSetSize2, std::size_t _ParParLabelSetSize, std::size_t _RootSize2>
    ExprNode(ExprNode<_ParLabelSetSize2,_ParParLabelSetSize,_RootSize2> *parent_) :
                            TnsNode<TnsSize>(),
                            // mParLabelSetSize(_ParLabelSetSize2),
                            // mParParLabelSetSize(_ParParLabelSetSize),
                            // mParOffset(reinterpret_cast<char*>(parent_) - reinterpret_cast<char*>(this)),
                            parent(parent_),
                            left(this),
                            right(this)
    {
      static_assert(_ParLabelSetSize2 == ParLabelSetSize, "Wrong expansion!");
      static_assert(_RootSize2         == RootSize,       "Wrong expansion!");
      static_assert(_ParLabelSetSize2 != 0,               "Wrong expansion!");
    }

  public:

    /**
     * @returns The @c mLabelSet member variable of @c ExprNode with the length 
     *          of each the @c Tensor 's dimensions. Needs explicit specification 
     *          for the @c stl array size @c TnsSize.
     * 
     * @returns The @c mLabelSet of @c ExprNode.
     */
    void *TnsDims()
    {
      return &mTnsDims;
    }
    
    /**
     * @returns The @c mTnsDims member variable of @c ExprNode with the set 
     *          of identification for @c TnsNode. Needs explicit
     *          specification for the @c stl @c array size @c LabelSetSize .
     */
    void *LabelSet()
    {
      return &mLabelSet;
    }

    /**
     * @returns The @c mDeltaSet member variable of @c ExprNode with the set of
     *          identification for the neighboor- brother @c TnsNode. Needs 
     *          explicit specification for the @c stl size @c BrotherLabelSetSize.
     */
    void *DeltaSet()
    {
      return &mDeltaSet;
    }

    /**
     * Search the @c ExprTree in order to find the @c ExprNode
     * with @c akey.
     * 
     * @param aKey [in] Searching key value.
     * 
     * @returns The @c ExprNode that has the searched key.
     */
    I_TnsNode *SearchKey (int const aKey)
    {
      auto length = LabelSetSize;
      if (length > 1)
      {
        if (aKey <= mKey)
          return Left()->SearchKey(aKey);
        else
          return Right()->SearchKey(aKey);
      }
      else
      {
        if (aKey == mKey)
          return this;
        else
          return nullptr;
      }
    }

    /**
     * @returns If the @c Exprode that calls the function is not the 
     *          root node, then the Parent @c ExprNode is returned.
     */
    I_TnsNode *Parent()   { return IsRoot ? nullptr  : parent; }
    /**
     * @returns If the @c Exprode that calls the function is not a 
     *          leaf node, then the Left @c ExprNode of the @c this
     *          node is returned.
     */
    I_TnsNode *Left  ()   { return IsLeaf ? nullptr  : &left;  }
    /**
     * @returns If the @c Exprode that calls the function is not a 
     *          leaf node, then the Right @c ExprNode of the @c this
     *          node is returned.
     */
    I_TnsNode *Right ()   { return IsLeaf ? nullptr  : &right; }

    /*
     * Computes the N mode product of a tensor with a matrix.
     * 
     * @tparam _LabelSetSize        Size of the LabelSet of @c this node.
     * @tparam _ParLabelSetSize     Size of the LabelSet of the parent node.
     * @tparam _RootSize            Size of the LabelSet of the root node.
     * @param  aFactor     [in]     Factor, of type @c FactorDimTree, to use 
     *                              for tree mode N product.
     * @param  aNumFactors [in]     Total number of factors.
     * @param  id          [in]     Identification of the updating factor @c aFactor.
     * @param  aGramian    [in,out] Gramian matrix of the node.
     * @param  aDeltaSet   [in,out] Label set of the brother node after the N-mode product.
     * @param  aTnsDims    [in,out] @c stl with the dimensions of the final Tensor.
     * 
     * @returns A @c TnsNode of @c ParTnsSize is returned.
     */
    Parent_Tensor_Type TreeMode_N_Product( FactorDimTree                       *const  aFactor,
                                           int                                  const  aNumFactors,
                                           int                                  const  id,
                                           std::array<int,ParTnsSize>           const &aTnsDims,
                                           Hessian_Type                               &aGramian,
                                           std::array<int,BrotherLabelSetSize>        &aDeltaSet     );

    /*
     * Interface of @c TTV product computation, between a tensor and a matrix.
     * @tparam _LabelSetSize        Size of the LabelSet of @c this node.
     * @tparam _ParLabelSetSize     Size of the LabelSet of the parent node.
     * @tparam _RootSize            Size of the LabelSet of the root node.
     * @param aFactor      [in]     Factor (of type @c FactorDimTree) to use for @c TTV product.
     * @param aNumFactors  [in]     Total number of factors.
     * @param id           [in]     Identification of the updating factor.
     * @param aDeltaSet    [in]     @c stl with the Label set of the brother @c ExprNode
     *                              after the @c TTV product of size @c BrotherLabelSetSize.
     * @param aX_partial   [in]     @c Tensor used for @c TTV product.
     * @param aTnsDims     [in,out] @c stl with the dimensions of the computed Tensor
     *                              of size @c ParTnsSize.
     * @param aGramian     [in,out] Gramian matrix of the @c ExprNode.
     * 
     * @returns A @c TnsNode with size equal to @c TnsSize. 
     */
    Tensor_Type TTVs                     ( FactorDimTree                       *const  aFactor,
                                           int                                  const  aNumFactors,
                                           int                                  const  id,
                                           std::array<int,BrotherLabelSetSize>  const &aDeltaSet,
                                           Parent_Tensor_Type                   const &aX_partial,
                                           std::array<int,ParTnsSize>           const &aTnsDims,
                                           Hessian_Type                               &aGramian     );

    /*
     * Computes the TTV product of a tensor with a matrix, using recursion.
     * 
     * @tparam _LabelSetSize         Size of the LabelSet of @c this node.
     * @tparam _ParLabelSetSize      Size of the LabelSet of the parent node.
     * @tparam _RootSize             Size of the LabelSet of the root node.
     * @tparam DeltaSetSize          Size of the DeltaSset of this node.
     * @tparam ResTnsSize            Order of the resulting @c Tensor.
     * @tparam ResParTnsSize         Order of the parent's @c Tensor.
     * @param  it           [in]     Factor (of @c FactorDimTree type) to use for @c TTV product.
     * @param  aX_partial   [in]     @c Tensor for @c TTV product.
     * @param  aContractDim [in]     Dimension for @c TTV product, based on being a Left 
     *                               or Right child.
     * @param  aTnsDims     [in,out] @c stl @c array with the dimensions of the computed Tensor
     *                               of size @c ResParTnsSize.
     * @param  aGramian     [in,out] Gramian matrix of the @c ExprNode.
     * @param  aX_result    [in,out] The result of TTV ( @c Tensor ) of size @c ResTnsSize.
     */
    template <std::size_t DeltaSetSize, std::size_t ResTnsSize, std::size_t ResParTnsSize>
    void       TTVs_util                 (  FactorDimTree                           *const  it,
                                            Tensor<static_cast<int>(ResParTnsSize)>  const &aX_partial,
                                            int                                      const  aContractDim,
                                            std::array<int,ResParTnsSize>            const &aTnsDims,
                                            Hessian_Type                                   &aGramian,
                                            Tensor<static_cast<int>(ResTnsSize)>           &aX_result   );

    /*
     * Updates the factors in each node until computing the leaf nodes and their 
     * @c Eigen @c Tensors. Based on the position of the node chooses to execute 
     * @c TreeMode_N_Product or @c TTV. Works in recursive way.
     * 
     * @tparam _LabelSetSize        Size of the LabelSet.
     * @tparam _ParLabelSetSize     Size of the LabelSet of the parent node.
     * @tparam _RootSize            Size of the LabelSet of the root node.
     * @param  aNumFactors [in]     Total number of factors.
     * @param  id          [in]     Identification of the updating factor.
     * @param  aFactor     [in,out] The factor to be updated.
     */
    void UpdateTree(int const aNumFactors, int const id, FactorDimTree *aFactor) override;
  };

  /*
   * typedef for zero order tensor.
   * @tparam TreeDim Tensor Order. 
   */
  template <std::size_t _TreeDim>
  using NullExprNode = ExprNode<0,1,_TreeDim>;

  /**
   * Container of the Dimension Tree.
   * @tparam TnsSize Tensor order.
   */
  template <std::size_t _TnsSize>
  struct ExprTree
  {
    static_assert(_TnsSize >= 1, "Expansion problem in ExprTree!");
    static constexpr std::size_t TnsSize = _TnsSize;
    static constexpr bool        IsNull  = false;    /**< Checks if TnsSize is greater than zero. */

    using  RootExprNode = ExprNode<TnsSize,0,TnsSize>;
    using  DataType     = typename RootExprNode::DataType;

    RootExprNode  root;   /**< Root node of the Dimension tree. */

    /**
     * Creates the root @c ExprNode, the left-right childs @c ExprNodes and calls
     * the @c overloaded @c Create if needed for more @c ExprNodes.
     * 
     * @param aLabelSet [in] LabelSet of the root @c ExprNode.
     * @param aTnsDims  [in] Dimensions of the initial Tensor.
     * @param R         [in] Rank of the factorization.
     * @param aTnsX     [in] Initial Tensor of @c Tensor type.
     */
    template<typename Array>
    void Create( std::array<int,RootExprNode::LabelSetSize>       &aLabelSet,
                 Array                                      const &aTnsDims,
                 int                                        const  R,
                 Tensor<static_cast<int>(TnsSize)>          const &aTnsX    )
    {
      constexpr bool IsLeaf = RootExprNode::IsLeaf;
      constexpr bool IsRoot = RootExprNode::IsRoot;

      static_assert(! (IsLeaf && IsRoot), "Tree expression with 1 dimension!" );

      constexpr std::size_t vHalf         = RootExprNode::DIM_HALF_SIZE;
      constexpr std::size_t vLabelSetSize = RootExprNode::LabelSetSize;

      root.mUpdated  = true;
      root.mTnsX     = aTnsX;
      std::copy(aTnsDims.begin(),                aTnsDims.begin()+TnsSize, root.mTnsDims.begin());
      root.mLabelSet = aLabelSet;
      root.mGramian.setConstant(1);

      // Create left child
      std::copy(aLabelSet.begin(),               aLabelSet.begin()+vHalf, root.left.mLabelSet.begin());
      std::copy(aLabelSet.end()-(TnsSize-vHalf), aLabelSet.end(),         root.left.mDeltaSet.begin());
      std::copy(aTnsDims.begin(),                aTnsDims.begin()+vHalf,  root.left.mTnsDims.begin()+1);
      root.left.mTnsDims.front() = R;
      RandomTensorGen(root.left.mTnsDims, root.left.mTnsX);
      
      // Create right child
      std::copy(aLabelSet.end()-(TnsSize-vHalf), aLabelSet.end(),                root.right.mLabelSet.begin());
      std::copy(aLabelSet.begin(),               aLabelSet.begin()+vHalf,        root.right.mDeltaSet.begin());
      std::copy(aTnsDims.begin()+vHalf,          aTnsDims.begin()+vLabelSetSize, root.right.mTnsDims.begin()+1);
      root.right.mTnsDims.front() = R;
      RandomTensorGen(root.right.mTnsDims, root.right.mTnsX);
      
      // Set key in root node
      root.mKey = root.left.mLabelSet[vHalf-1];

      Create(root.left.mTnsDims,  R, root.left);
      Create(root.right.mTnsDims, R, root.right);
    }

    /**
     * Creates the @c ExprNodes, expcept the root and its children @c ExprNodes.
     * 
     * @param aTnsDims [in] Dimensions of the initial Tensor.
     * @param R        [in] Rank of the factorization.
     * @param expr     [in] Newly Created @c ExprNode.
     */
    template<typename Array, typename ExprNode>
    void Create( Array                const &aTnsDims, 
                 [[maybe_unused]] int const  R,      // not used in leaf nodes.
                 ExprNode                   &expr    )
    {
      static_assert(!std::is_same_v<ExprNode,NullExprNode<TnsSize>>, "Expansion problem!");

      using Expr_Node_Type = ExprNode;

      constexpr  bool IsLeaf              = Expr_Node_Type::IsLeaf;
      constexpr  std::size_t vExprTnsSize = Expr_Node_Type::TnsSize;

      if constexpr (!IsLeaf)
      {
        constexpr std::size_t vHalf         = Expr_Node_Type::DIM_HALF_SIZE;
        constexpr std::size_t vLabelSetSize = Expr_Node_Type::LabelSetSize;

        // Create left child
        std::copy(expr.mLabelSet.begin(),                      expr.mLabelSet.begin()+vHalf, expr.left.mLabelSet.begin());
        std::copy(expr.mLabelSet.end()-(vExprTnsSize-vHalf-1), expr.mLabelSet.end(),         expr.left.mDeltaSet.begin());
        std::copy(aTnsDims.begin(),                            aTnsDims.begin()+vHalf+1,     expr.left.mTnsDims.begin());
        RandomTensorGen(expr.left.mTnsDims, expr.left.mTnsX);
        
        // Create right child
        std::copy(expr.mLabelSet.end()-(vExprTnsSize-vHalf-1), expr.mLabelSet.end(),           expr.right.mLabelSet.begin());
        std::copy(expr.mLabelSet.begin(),                      expr.mLabelSet.begin()+vHalf,   expr.right.mDeltaSet.begin());
        std::copy(aTnsDims.begin()+vHalf+1,                    aTnsDims.begin()+vLabelSetSize+1, expr.right.mTnsDims.begin()+1);
        expr.right.mTnsDims.front() = R;
        RandomTensorGen(expr.right.mTnsDims,   expr.right.mTnsX);
        
        expr.mKey = expr.left.mLabelSet[vHalf-1];

        Create(expr.left.mTnsDims,  R, expr.left);
        Create(expr.right.mTnsDims, R, expr.right);
      }
      else
      {
        expr.mKey = expr.mLabelSet[0];
      }
    }

  };

  /*
   * ExprTree with no nodes.
   */
  template <>
  struct ExprTree<0>
  {
    static constexpr bool IsNull = true;   /**< Indicates that there are no data in this ExprTree. */
  };

  /**
   * Information about Factors and the associated leafs.
   * @c Factor is created in @c Tensor.hpp.
   */
  struct FactorDimTree : public Factor<Tensor<2>>
  {
      TnsNode<1>        *leaf;

      FactorDimTree() : leaf(nullptr)
      { }
  };

  /**
   * Computes the N mode product of a tensor with a matrix.
   * @tparam _LabelSetSize        Size of the LabelSet of @c this node.
   * @tparam _ParLabelSetSize     Size of the LabelSet of the parent node.
   * @tparam _RootSize            Size of the LabelSet of the root node.
   * @param  aFactor     [in]     Factor (of type @c FactorDimTree) to use 
   *                              for tree mode N product.
   * @param  aNumFactors [in]     Total number of factors.
   * @param  id          [in]     Indexing for the updating factor, @c aFactor.
   * @param  aGramian    [in,out] Gramian matrix of the node.
   * @param  aDeltaSet   [in,out] Label set of the brother node after the N-mode product.
   * @param  aTnsDims    [in,out] @c stl with the dimensions of the final Tensor.
   * 
   * @returns A @c TnsNode with size equal to @c ParTnsSize.
   */
  template <std::size_t _LabelSetSize, std::size_t _ParLabelSetSize, std::size_t _RootSize>
  typename ExprNode<_LabelSetSize,_ParLabelSetSize,_RootSize>::Parent_Tensor_Type
  ExprNode<_LabelSetSize,_ParLabelSetSize,_RootSize>::TreeMode_N_Product( FactorDimTree                      *const  aFactor,
                                                                          int                                 const  aNumFactors,
                                                                          int                                 const  id,
                                                                          std::array<int,ParTnsSize>          const &aTnsDims,
                                                                          Hessian_Type                              &aGramian,
                                                                          std::array<int,BrotherLabelSetSize>       &aDeltaSet     )
  {
    int                    vContractDim;
    Parent_Tensor_Type     vX_partial;
    FactorDimTree         *it;
    constexpr std::size_t  aDeltaSetSize = BrotherLabelSetSize-1;

    static_assert(!IsRoot, "TreeMode_N_Product() must not be called on root node!");

    const std::size_t                    R = aFactor->gramian.dimension(0);
    std::array<int,ParTnsSize>           vTnsDims;
    std::array<Eigen::IndexPair<int>, 1> product_dims;
    
    if (this == Parent()->Left())
    {
      it           = aFactor+aNumFactors-id-1;
      vContractDim = RootSize-1;
      std::copy(mDeltaSet.begin(), mDeltaSet.begin()+aDeltaSetSize, aDeltaSet.begin());

      std::copy(aTnsDims.begin(), aTnsDims.end()-1, vTnsDims.begin()+1);
      vTnsDims.front() = R;
    }
    else
    {
      it           = aFactor-id;
      vContractDim = 0;
      std::copy(mDeltaSet.end()-aDeltaSetSize, mDeltaSet.end(), aDeltaSet.begin());
      std::copy(aTnsDims.begin()+1, aTnsDims.end(), vTnsDims.begin()+1);
      vTnsDims.front() = R;
    }

    aGramian     = (*it).gramian;
    product_dims = { Eigen::IndexPair<int>(0, vContractDim) };

    vX_partial.resize(vTnsDims);
    vX_partial = it->factor.contract(*reinterpret_cast<Parent_Tensor_Type *>(Parent()->TensorX()), product_dims);

    return vX_partial;
  }

  /**
   * Computes the TTV product of a tensor with a matrix, using recursion.
   * @tparam _LabelSetSize         Size of the LabelSet of @c this node.
   * @tparam _ParLabelSetSize      Size of the LabelSet of the parent node.
   * @tparam _RootSize             Size of the LabelSet of the root node.
   * @tparam DeltaSetSize          Size of the DeltaSset of this node.
   * @tparam ResTnsSize            Order of the resulting @c Tensor.
   * @tparam ResParTnsSize         Order of the parent's @c Tensor.
   * @param  it           [in]     Factor (of @c FactorDimTree type) to use for @c TTV product.
   * @param  aX_partial   [in]     @c Tensor for @c TTV product.
   * @param  aContractDim [in]     Dimension for @c TTV product, based on being a Left 
   *                               or Right child.
   * @param  aTnsDims     [in,out] @c stl @c array with the dimensions of the computed Tensor
   *                               of size @c ResParTnsSize.
   * @param  aGramian     [in,out] Gramian matrix of the @c ExprNode.
   * @param  aX_result    [in,out] The result of TTV @c Tensor of size @c ResTnsSize.
   */
  template <std::size_t _LabelSetSize, std::size_t _ParLabelSetSize, std::size_t _RootSize>
  template <std::size_t DeltaSetSize, std::size_t ResTnsSize, std::size_t ResParTnsSize>
  void
  ExprNode<_LabelSetSize,_ParLabelSetSize,_RootSize>::TTVs_util(  FactorDimTree                           *const  it,
                                                                  Tensor<static_cast<int>(ResParTnsSize)>  const &aX_partial,
                                                                  int                                      const  aContractDim,
                                                                  std::array<int,ResParTnsSize>            const &aTnsDims,
                                                                  Hessian_Type                                   &aGramian,
                                                                  Tensor<static_cast<int>(ResTnsSize)>           &aX_result   )
  {
    constexpr int _ResParTnsSize = ResParTnsSize - 1;

    using Result_Tensor_Type = Tensor<_ResParTnsSize>;

    std::array<int,static_cast<std::size_t>(_ResParTnsSize)> vTnsDims;
    const std::size_t R = aGramian.dimension(0);

    aGramian *= (*it).gramian;   // Hadamard Product.
    
    // Allocate the reduction result
    if (this == Parent()->Right())  // Right Child
    {
      std::copy(aTnsDims.end()-_ResParTnsSize+1, aTnsDims.end(), vTnsDims.begin()+1);
      vTnsDims.front() = R;
    }
    else                            // Left Child
    {
      // 1o TTV
      if constexpr (IsFirstChild)
      {
        // First Child and First TTV
        if constexpr (DeltaSetSize == BrotherLabelSetSize - 1)
        {
          std::copy(aTnsDims.begin(), aTnsDims.begin()+_LabelSetSize, vTnsDims.begin()+1);
          if constexpr (_ResParTnsSize - _LabelSetSize > 0)
            std::copy(aTnsDims.end() - (_ResParTnsSize - _LabelSetSize), aTnsDims.end(), vTnsDims.begin()+_LabelSetSize+1);
          vTnsDims.front() = R;
        }
        else
        {
          std::copy(aTnsDims.begin(), aTnsDims.begin()+_LabelSetSize+1, vTnsDims.begin());
          if constexpr (_ResParTnsSize - _LabelSetSize > 0)
              std::copy(aTnsDims.end() - (_ResParTnsSize - _LabelSetSize) + 1, aTnsDims.end(), vTnsDims.begin()+_LabelSetSize+1);
        }
      }
      else
      {
        std::copy(aTnsDims.begin(), aTnsDims.begin()+_LabelSetSize+1, vTnsDims.begin());
        if constexpr (_ResParTnsSize - _LabelSetSize > 0)
            std::copy(aTnsDims.end() - (_ResParTnsSize - _LabelSetSize) + 1, aTnsDims.end(), vTnsDims.begin()+_LabelSetSize+1);
      }
    }

    // Apply reduction
    Result_Tensor_Type  vTnsX;
    vTnsX.resize(vTnsDims);
    
    TensorPartialProduct_R<ResParTnsSize,_ResParTnsSize>(aX_partial, it->factor, 0, aContractDim, &vTnsX);

    // Update tensor orders
    if constexpr (_ResParTnsSize > ResTnsSize)
    {
      TTVs_util<DeltaSetSize-1,ResTnsSize>(it+1, vTnsX, aContractDim, vTnsDims, aGramian, aX_result);
    }
    else
    {
      aX_result = vTnsX;
    }
  }

  /**
   * Interface of @c TTV product computation, between a tensor and a matrix.
   * @tparam _LabelSetSize        Size of the LabelSet of @c this node.
   * @tparam _ParLabelSetSize     Size of the LabelSet of the parent node.
   * @tparam _RootSize            Size of the LabelSet of the root node.
   * @param aFactor      [in]     Factor (of type @c FactorDimTree) to use for @c TTV product.
   * @param aNumFactors  [in]     Total number of factors.
   * @param id           [in]     Identification of the updating factor.
   * @param aDeltaSet    [in]     @c stl with the Label set of the brother @c ExprNode
   *                              after the @c TTV product of size @c BrotherLabelSetSize.
   * @param aX_partial   [in]     @c Eigen @c Tensor used for @c TTV product.
   * @param aTnsDims     [in,out] @c stl with the dimensions of the computed Tensor
   *                              of size @c ParTnsSize.
   * @param aGramian     [in,out] Gramian matrix of the @c ExprNode.
   * 
   * @returns An @c TnsNode with size equal to @c TnsSize. 
   */
  template <std::size_t _LabelSetSize, std::size_t _ParLabelSetSize, std::size_t _RootSize>
  typename ExprNode<_LabelSetSize,_ParLabelSetSize,_RootSize>::Tensor_Type
  ExprNode<_LabelSetSize,_ParLabelSetSize,_RootSize>::TTVs( FactorDimTree                       *const  aFactor,
                                                            int                                  const  aNumFactors,
                                                            int                                  const  id,
                                                            std::array<int,BrotherLabelSetSize>  const &aDeltaSet,
                                                            Parent_Tensor_Type                   const &aX_partial,
                                                            std::array<int,ParTnsSize>           const &aTnsDims,
                                                            Hessian_Type                               &aGramian     )
  {
    using Result_Tensor_Type = ExprNode<_LabelSetSize,_ParLabelSetSize,_RootSize>::Tensor_Type;

    constexpr std::size_t vDeltaSetSize = IsFirstChild ? BrotherLabelSetSize - 1 : BrotherLabelSetSize;

    static_assert( (vDeltaSetSize+TnsSize) == ParTnsSize, "Wrong call!" );

    int  vContractDim = (this == Parent()->Right()) ? 0 : LabelSetSize;

    FactorDimTree *it;

    it = aFactor;               // TODO  There is no range check for it !
    assert(aDeltaSet[0] - 1 < aNumFactors);
    it = aFactor + (aDeltaSet[0] - id - 1);

    Result_Tensor_Type vResTensor;

    TTVs_util<vDeltaSetSize,TnsSize>(it, aX_partial, vContractDim, aTnsDims, aGramian, vResTensor);

    return vResTensor;
  }

  /**
   * Updates the factors in each node until computing the leaf nodes and their 
   * @c Tensors. Based on the position of the node chooses to execute 
   * @c TreeMode_N_Product or @c TTV. Works in recursive way.
   * 
   * @tparam _LabelSetSize        Size of the LabelSet.
   * @tparam _ParLabelSetSize     Size of the LabelSet of the parent node.
   * @tparam _RootSize            Size of the LabelSet of the root node.
   * @param  aNumFactors [in]     Total number of factors.
   * @param  id          [in]     Identification of the updating factor.
   * @param  aFactor     [in,out] The factor to be updated.
   */
  template <std::size_t _LabelSetSize, std::size_t _ParLabelSetSize, std::size_t _RootSize>
  void ExprNode<_LabelSetSize,_ParLabelSetSize,_RootSize>::UpdateTree(int const aNumFactors, int const id, FactorDimTree *aFactor)
  {
    if (mUpdated)  // Root is always mUpdated == true
    {
      if constexpr (!IsRoot)
        left.SetOutdated();
      else
      {
        if constexpr (!IsLeaf)
        {
          if (left.mUpdated)
            left.SetOutdated();
          else if (right.mUpdated)
            right.SetOutdated();
        }
      }
    }
    else
    {    
      Parent()->UpdateTree(aNumFactors, id, aFactor);
      int R   = mTnsDims.back();

      Hessian_Type                         vGramian(R,R);
      Tensor_Type                          vX_temp;
      Parent_Tensor_Type                   vX_partial;

      std::array<int, ParTnsSize>          vTnsDims;
      std::array<int, BrotherLabelSetSize> vDeltaSet;

      std::size_t vDeltaSetSize = (IsFirstChild) ? BrotherLabelSetSize-1 : BrotherLabelSetSize;

      vTnsDims = *reinterpret_cast<std::array<int, ParTnsSize> *>(Parent()->TnsDims());

      if constexpr (IsFirstChild)
      {  // Case: leaf is a child of the root: mode n product is required
        vX_partial = TreeMode_N_Product(aFactor, aNumFactors, id, vTnsDims, vGramian, vDeltaSet);
      }
      else
      {
        std::copy(mDeltaSet.begin(), mDeltaSet.begin()+vDeltaSetSize, vDeltaSet.begin());
        vX_partial = *reinterpret_cast<Parent_Tensor_Type *>(Parent()->TensorX());
        vGramian   = Parent()->Gramian();
      }

      if constexpr (TnsSize == ParTnsSize)
      {
        vX_temp = vX_partial;
      }
      else if constexpr (!IsRoot)
      {
        if (vDeltaSetSize > 0)
        {
          vX_temp = TTVs(aFactor, aNumFactors, id, vDeltaSet, vX_partial, vTnsDims, vGramian); // TODO check for bad allocation
        }   // TODO check all paths
      }
      else
      {
        // TODO Now???
      }

      // Update the current tree node
      mTnsX    = vX_temp;
      mGramian = vGramian;
      mUpdated = true;
    }
  }

  /**
   * Searches in whole @c tree to find the @c ExprNode with the
   * @c key. Make use of @c SearchKey.
   * 
   * @param key  [in]     Searching key value
   * @param tree [in,out] Expression tree
   * 
   * @returns The @c ExprNode in the specified @c tree, that has @c key.
   */
  template<std::size_t _TreeDim>
  I_TnsNode *search_leaf(int const key, ExprTree<_TreeDim> &tree)
  {
    if constexpr (_TreeDim >= 0)
      return tree.root.SearchKey(key);
    else
      return nullptr;
  }

} // end namespace partensor

#endif // end of DIM_TREES_HPP
