


        st.caption("Showing the most frequent categories.")

    else:

        st.info("No categorical features are available.")



# ---------- CORRELATION ----------

with tab3:

    if len(numeric_cols) >= 2:

        corr = data[numeric_cols].corr()



        heatmap_height = min(

            max(chart_height, len(numeric_cols) * 0.35),

            8

        )



        fig, ax = plt.subplots(

            figsize=(chart_width, heatmap_height)

        )



        image = ax.imshow(

            corr,

            aspect="auto",

            vmin=-1,

            vmax=1

        )



        ax.set_xticks(range(len(numeric_cols)))

        ax.set_yticks(range(len(numeric_cols)))



        ax.set_xticklabels(

            numeric_cols,

            rotation=45,

            ha="right",

            fontsize=7

        )



        ax.set_yticklabels(numeric_cols, fontsize=7)

        ax.set_title("Numeric Feature Correlations")



        fig.colorbar(image, ax=ax, label="Correlation")

        fig.tight_layout()



        st.pyplot(fig, use_container_width=False)

        plt.close(fig)

    else:

        st.info("At least two numeric features are needed for correlation.")



# ---------- DYNAMIC QUESTION SUGGESTIONS ----------

st.subheader("Ask Your Data")

st.caption("Choose a suggestion or type your own question.")



suggestions = [

    "How many rows and columns are in the dataset?",

    "Are there any missing values?"

]



if numeric_cols:

    suggestions.append(

        f"What are the statistics for {numeric_cols[0]}?"

    )

    suggestions.append(

        f"What is the highest {numeric_cols[0]}?"

    )



if categorical_cols:

    suggestions.append(

        f"What are the most common values in {categorical_cols[0]}?"

    )



if numeric_cols and categorical_cols:

    suggestions.append(

        f"What is the average {numeric_cols[0]} by {categorical_cols[0]}?"

    )

elif len(numeric_cols) >= 2:

    suggestions.append(

        f"What is the correlation between {numeric_cols[0]} and {numeric_cols[1]}?"

    )



# Display no more than six unique suggestions

suggestions = list(dict.fromkeys(suggestions))[:6]

