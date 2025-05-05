import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import "./Products.css"


const Products = (props) => {
  const [products, setProducts] = useState(props.products);

  useEffect(() => {
    setProducts(props.products);
  }, [props.products]);

  return (
    <div style={{ maxWidth: '500px', margin: 'auto' }}>
      <div style={{ minHeight: '300px', maxHeight: '500px', padding: '1rem', marginBottom: '4.6rem' }}>
        <ReactMarkdown components={{a: ({node, ...props}) => (<a {...props} target="_blank" rel="noopener noreferrer"/>)}}>{products}</ReactMarkdown>
    </div>
    </div>
  );
};

export default Products;
